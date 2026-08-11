#!/usr/bin/env python3
"""Packs molecules into a cubic simulation cell with Packmol.

Feed it PDB files (from Avogadro or wherever you got them) and it packs them
into a periodic cubic box, either at explicit counts or a target molarity.

Needs the `packmol` binary on PATH.
Install:  conda install -c conda-forge packmol
     or:  sudo apt install packmol

Examples
--------
python pack_cell.py \
  -m Na:inputs/Na.pdb:0.5M \
  -m PF6:inputs/PF6.pdb:0.5M \
  -m DME:inputs/DME.pdb:200 \
  --box-size 30 \
  --output packed/system.pdb

python pack_cell.py --project ./my-project \
  -m Na:inputs/Na.pdb:0.5M \
  -m PF6:inputs/PF6.pdb:0.5M \
  -m DME:inputs/DME.pdb:200 \
  --box-size 30
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from utils import parse_molecule_spec, add_cryst1_to_pdb, ProjectLayout


def write_packmol_input(molecules, box_size, output_path, tolerance, seed):
    """Builds the packmol input as a string, ready to pipe in. Doesn't touch disk."""
    lines = [
        f"tolerance {tolerance}",
        "filetype pdb",
        f"output {output_path}",
    ]
    if seed is not None:
        lines.append(f"seed {seed}")
    lines.append("")

    for name, path, count in molecules:
        lines.append(f"structure {path}")
        lines.append(f"  number {count}")
        lines.append(f"  inside box 0. 0. 0. {box_size} {box_size} {box_size}")
        lines.append("end structure")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Pack molecules into a cubic simulation cell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project",
                        help="Project root, output defaults to <project>/packed/system.pdb.")
    parser.add_argument("-m", "--molecule", action="append", required=True,
                        metavar="NAME:PATH:AMOUNT",
                        help="'name:path:amount' (amount is an int or float+M, e.g. 1.0M).")
    parser.add_argument("-b", "--box-size", type=float, required=True,
                        help="Cubic box edge length in angstroms.")
    parser.add_argument("-o", "--output",
                        help="Output PDB file path.")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Min distance between atoms in Å (default: 2.0).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the Packmol input without executing.")

    args = parser.parse_args()

    if args.project:
        layout = ProjectLayout(args.project)
        args.output = args.output or layout.packed_pdb

    if not args.output:
        parser.error("Provide --project or --output.")

    if shutil.which("packmol") is None:
        print("ERROR: 'packmol' not found on PATH.")
        print("  Install: conda install -c conda-forge packmol")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    molecules = []
    print(f"Box size: {args.box_size:.1f} Å  ({args.box_size**3:.0f} Å^3)")
    print(f"Volume:   {args.box_size**3 * 1e-27:.4e} L\n")

    for spec in args.molecule:
        name, path, count = parse_molecule_spec(spec, args.box_size)
        if not os.path.isfile(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        molecules.append((name, path, count))
        print(f"  {name:>10s}: {count:>5d} molecules  ({path})")

    total_molecules = sum(c for _, _, c in molecules)
    print(f"\n  {'Total':>10s}: {total_molecules:>5d} molecules")

    inp_content = write_packmol_input(
        molecules, args.box_size, os.path.abspath(args.output),
        args.tolerance, args.seed,
    )

    if args.dry_run:
        print("\n--- Packmol input ---")
        print(inp_content)
        return

    print("\nRunning packmol...")
    result = subprocess.run(
        ["packmol"],
        input=inp_content,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not os.path.exists(args.output):
        print(f"ERROR: Packmol failed (exit code {result.returncode})")
        print(result.stdout[-500:] if result.stdout else "")
        print(result.stderr[-500:] if result.stderr else "")
        sys.exit(1)

    for line in result.stdout.strip().split("\n")[-3:]:
        print(f"  {line}")

    print(f"\nAdding CRYST1 record ({args.box_size:.3f} Å cubic cell)...")
    add_cryst1_to_pdb(args.output, args.box_size)

    print(f"Packed cell written to: {args.output}")


if __name__ == "__main__":
    main()
