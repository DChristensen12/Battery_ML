#!/usr/bin/env python3
"""Converts an ASE trajectory into something VMD can actually open (XYZ or multi-model PDB).

Reads trajectory.traj from run_md.py and writes it back out.

Requirements:
    pip install ase numpy

Examples
--------
python export_vmd.py --project ./my-project --protocol npt
python export_vmd.py --project ./my-project --protocol npt --stride 10
python export_vmd.py -t npt/trajectory.traj -o vmd/trajectory.xyz
python export_vmd.py -t npt/trajectory.traj -o vmd/trajectory.pdb -f pdb
"""

import argparse
import os

from utils import ProjectLayout


def main():
    parser = argparse.ArgumentParser(
        description="Convert ASE trajectory to VMD-readable XYZ or PDB.",
    )
    parser.add_argument("--project",
                        help="Project root, auto-derives paths.")
    parser.add_argument("--protocol", default="npt",
                        choices=["nvt", "npt", "anneal"],
                        help="Which equilibration to export (with --project).")
    parser.add_argument("-t", "--trajectory",
                        help="Path to trajectory.traj.")
    parser.add_argument("-o", "--output",
                        help="Output file (.xyz or .pdb).")
    parser.add_argument("-f", "--format", choices=["xyz", "pdb"], default=None,
                        help="Output format (auto-detected from extension).")
    parser.add_argument("-s", "--stride", type=int, default=1,
                        help="Take every Nth frame (default: 1).")

    args = parser.parse_args()

    if args.project:
        layout = ProjectLayout(args.project)
        args.trajectory = args.trajectory or layout.trajectory(args.protocol)
        fmt_ext = args.format or "xyz"
        args.output = args.output or layout.vmd_trajectory(fmt_ext)

    if not args.trajectory or not args.output:
        parser.error("Provide --project, or both --trajectory and --output.")

    fmt = args.format
    if fmt is None:
        ext = os.path.splitext(args.output)[1].lower()
        fmt = "pdb" if ext == ".pdb" else "extxyz"
    elif fmt == "xyz":
        fmt = "extxyz"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    from ase.io import write as ase_write
    from ase.io.trajectory import Trajectory

    print(f"Reading {args.trajectory}...")
    traj = Trajectory(args.trajectory, "r")
    n_total = len(traj)
    print(f"  {n_total} frames")

    frames = [traj[i] for i in range(0, n_total, args.stride)]
    n_out = len(frames)
    traj.close()

    print(f"Writing {n_out} frames to {args.output} ({fmt})...")
    ase_write(args.output, frames, format=fmt)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")
    print(f"\nTo view: vmd {os.path.basename(args.output)}")


if __name__ == "__main__":
    main()
