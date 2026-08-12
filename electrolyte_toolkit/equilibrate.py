#!/usr/bin/env python3
"""Electrolyte box equilibration with persistent local checkpointing.

Supports multiple workflows and potentials:

  Workflows:
    nvt+npt   NVT thermalization then NPT density relaxation (Dataset 1 style)
    npt       NPT only, no prior NVT (Dataset 3 / FAIRChem style)
    nvt       NVT only, no NPT

  Potentials:
    orbmol_v2   OrbMol-v2 (Orbital Materials), trained on OMol25
    uma         UMA-s-1.2 (FAIRChem/Meta), trained on OMol25

Checkpoints are saved to a local folder. Only the latest checkpoint per
box is kept. If interrupted, rerun the same command to resume automatically.

Outputs go to: <checkpoint_dir>/<box_name>/
    ckpt_<phase>_<step>.xyz   latest checkpoint structure
    ckpt_<phase>_<step>.npz   latest checkpoint logs
    <box_name>.xyz            final equilibrated structure
    diagnostics.png           T, PE, density vs time

Examples
--------
# Dataset 3 style: NPT-only with UMA
python equilibrate.py my_box.xyz --model uma --workflow npt

# Dataset 1 style: NVT+NPT with OrbMol-v2
python equilibrate.py my_box.xyz --model orbmol_v2 --workflow nvt+npt

# NVT only with OrbMol-v2
python equilibrate.py my_box.xyz --model orbmol_v2 --workflow nvt

# Custom steps
python equilibrate.py my_box.xyz --model uma --workflow npt --npt-steps 50000

# Resume an interrupted run
python equilibrate.py --resume my_box_name

# Check status of all boxes
python equilibrate.py --status
"""

import argparse
import glob
import os
import sys
import time

import numpy as np


DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), "electrolyte_equilibration")
NVT_STEPS = 50000      # 50 ps
NPT_STEPS = 100000     # 100 ps
CHECKPOINT_EVERY = 5000
LOG_EVERY = 100
TEMPERATURE = 300.0     # K

try:
    from utils import get_calculator as _get_calc, ATM_TO_EV_A3
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import get_calculator as _get_calc, ATM_TO_EV_A3


NPT_PARAMS = {
    "orbmol_v2": {
        "externalstress_eV_A3": 0.000101325 * 0.006242,
        "ttime_fs": 25.0,
        "pfactor_val": lambda fs: (75.0 * fs) ** 2 * 1.0 * 0.006242,
        "mask": None,
    },
    "uma": {
        "externalstress_bar": 1.0,
        "ttime_fs": 100.0,
        "pfactor_val": lambda fs: 0.1,
        "mask": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    },
}


def get_npt_kwargs(model, units_mod):
    """Return the NPT constructor kwargs for the given model."""
    if model.startswith("uma"):
        p = NPT_PARAMS["uma"]
        kwargs = dict(
            externalstress=p["externalstress_bar"] * units_mod.bar,
            ttime=p["ttime_fs"] * units_mod.fs,
            pfactor=p["pfactor_val"](units_mod.fs),
            mask=p["mask"],
        )
    else:
        p = NPT_PARAMS["orbmol_v2"]
        kwargs = dict(
            externalstress=p["externalstress_eV_A3"],
            ttime=p["ttime_fs"] * units_mod.fs,
            pfactor=p["pfactor_val"](units_mod.fs),
        )
    return kwargs


def box_dir(checkpoint_dir, name):
    d = os.path.join(checkpoint_dir, name)
    os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(atoms, checkpoint_dir, name, step, phase, temps, pes, densities=None):
    from ase.io import write
    d = box_dir(checkpoint_dir, name)
    for f in glob.glob(os.path.join(d, "ckpt_*.xyz")):
        os.remove(f)
    for f in glob.glob(os.path.join(d, "ckpt_*.npz")):
        os.remove(f)

    write(os.path.join(d, f"ckpt_{phase}_{step}.xyz"), atoms, format="extxyz")
    save_data = dict(step=step, phase=phase, temps=np.array(temps), pes=np.array(pes))
    if densities is not None:
        save_data["densities"] = np.array(densities)
    np.savez(os.path.join(d, f"ckpt_{phase}_{step}.npz"), **save_data)

    rho_str = f", rho={densities[-1]:.4f}" if densities else ""
    print(f"    [CHECKPOINT] {phase} step {step}{rho_str} -> {d}", flush=True)


def load_checkpoint(checkpoint_dir, name):
    from ase.io import read
    d = box_dir(checkpoint_dir, name)
    xyzs = sorted(glob.glob(os.path.join(d, "ckpt_*.xyz")))
    npzs = sorted(glob.glob(os.path.join(d, "ckpt_*.npz")))
    if not xyzs or not npzs:
        return None, 0, None, [], [], []
    atoms = read(xyzs[-1])
    data = np.load(npzs[-1], allow_pickle=True)
    densities = data["densities"].tolist() if "densities" in data else []
    return (atoms, int(data["step"]), str(data["phase"]),
            data["temps"].tolist(), data["pes"].tolist(), densities)


def check_convergence(densities, threshold_pct=1.0):
    n = len(densities)
    if n < 100:
        return False, float("inf"), 0.0
    window = n // 5
    prev = densities[-(2 * window):-window]
    last = densities[-window:]
    mean_prev = np.mean(prev)
    mean_last = np.mean(last)
    drift = 100 * abs(mean_last - mean_prev) / mean_last
    return drift < threshold_pct, drift, mean_last


def run_nvt(atoms, name, checkpoint_dir, model, n_steps=NVT_STEPS, start_step=0,
            prev_temps=None, prev_pes=None, device="cuda"):
    import torch
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units

    atoms.pbc = True
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    atoms.calc = _get_calc(model, device=device)

    if start_step == 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)

    dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=TEMPERATURE,
                   friction=0.01 / units.fs)
    dyn.nsteps = start_step

    temps = list(prev_temps or [])
    pes = list(prev_pes or [])
    remaining = n_steps - start_step
    t0 = time.time()

    print(f"  {len(atoms)} atoms | {remaining} steps remaining", flush=True)

    def logger():
        T = atoms.get_temperature()
        pe = atoms.get_potential_energy() / len(atoms)
        temps.append(T); pes.append(pe)
        step = dyn.nsteps
        if step % 1000 == 0:
            elapsed = time.time() - t0
            rate = (step - start_step) / elapsed if elapsed > 0 else 0
            eta = (n_steps - step) / rate / 60 if rate > 0 else 0
            print(f"    step {step:6d}/{n_steps}  T={T:.1f}K  "
                  f"PE={pe:.4f}  [{rate:.1f} st/s, ETA {eta:.0f}m]", flush=True)

    def checkpointer():
        step = dyn.nsteps
        if step > start_step and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(atoms, checkpoint_dir, name, step, "nvt", temps, pes)

    dyn.attach(logger, interval=LOG_EVERY)
    dyn.attach(checkpointer, interval=LOG_EVERY)

    if remaining <= 0:
        print("  Already complete.", flush=True)
        return atoms, temps, pes

    dyn.run(remaining)
    save_checkpoint(atoms, checkpoint_dir, name, n_steps, "nvt", temps, pes)
    return atoms, temps, pes


def run_npt(atoms, name, checkpoint_dir, model, n_steps=NPT_STEPS, start_step=0,
            prev_temps=None, prev_pes=None, prev_densities=None, device="cuda"):
    import torch
    from ase.io import write
    from ase.md.melchionna import MelchionnaNPT as NPT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units

    atoms.pbc = True
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    total_mass = sum(atoms.get_masses())

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    atoms.calc = _get_calc(model, device=device)

    if start_step == 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)

    npt_kwargs = get_npt_kwargs(model, units)
    dyn = NPT(atoms, timestep=1.0 * units.fs, temperature_K=TEMPERATURE, **npt_kwargs)
    dyn.nsteps = start_step

    temps = list(prev_temps or [])
    pes = list(prev_pes or [])
    densities = list(prev_densities or [])
    remaining = n_steps - start_step
    t0 = time.time()

    rho0 = total_mass / atoms.get_volume() * 1.66054
    print(f"  {len(atoms)} atoms | starting rho={rho0:.4f} g/cm3 | "
          f"{remaining} steps remaining", flush=True)

    def logger():
        T = atoms.get_temperature()
        pe = atoms.get_potential_energy() / len(atoms)
        rho = total_mass / atoms.get_volume() * 1.66054
        temps.append(T); pes.append(pe); densities.append(rho)
        step = dyn.nsteps
        if step % 1000 == 0:
            elapsed = time.time() - t0
            rate = (step - start_step) / elapsed if elapsed > 0 else 0
            eta = (n_steps - step) / rate / 60 if rate > 0 else 0
            conv_str = ""
            converged, drift, mean_rho = check_convergence(densities)
            if converged:
                conv_str = f" ** CONVERGED (drift={drift:.2f}%, mean={mean_rho:.4f}) **"
            elif drift < float("inf"):
                conv_str = f" (drift={drift:.1f}%)"
            print(f"    step {step:6d}/{n_steps}  T={T:.1f}K  rho={rho:.4f}  "
                  f"PE={pe:.4f}  [{rate:.1f} st/s, ETA {eta:.0f}m]{conv_str}", flush=True)

    def checkpointer():
        step = dyn.nsteps
        if step > start_step and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(atoms, checkpoint_dir, name, step, "npt",
                            temps, pes, densities)

    dyn.attach(logger, interval=LOG_EVERY)
    dyn.attach(checkpointer, interval=LOG_EVERY)

    if remaining <= 0:
        print("  Already complete.", flush=True)
        return atoms, temps, pes, densities

    dyn.run(remaining)

    save_checkpoint(atoms, checkpoint_dir, name, n_steps, "npt", temps, pes, densities)
    final_path = os.path.join(box_dir(checkpoint_dir, name), f"{name}.xyz")
    write(final_path, atoms, format="extxyz")

    rho_f = total_mass / atoms.get_volume() * 1.66054
    has_nan = np.any(np.isnan(atoms.get_positions()))
    if has_nan:
        print("  WARNING: NaN in final positions!", flush=True)
    else:
        converged, drift, mean_rho = check_convergence(densities)
        status = "CONVERGED" if converged else f"NOT CONVERGED (drift={drift:.1f}%)"
        print(f"  DONE: rho {rho0:.4f} -> {rho_f:.4f} g/cm3 | {status}", flush=True)
        print(f"  Final: {final_path}", flush=True)

    return atoms, temps, pes, densities


def plot_diagnostics(name, checkpoint_dir, temps, pes, densities=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(temps) == 0:
        print("No data to plot.")
        return

    n_plots = 3 if densities else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    t_ps = np.arange(len(temps)) * LOG_EVERY * 0.001
    w = min(50, max(1, len(temps) // 4))

    axes[0].plot(t_ps, temps, alpha=0.3, lw=0.5)
    if w > 1:
        axes[0].plot(t_ps[w - 1:], np.convolve(temps, np.ones(w) / w, "valid"), "r", lw=1.5)
    axes[0].axhline(300, color="k", ls="--", alpha=0.4)
    axes[0].set(xlabel="Time (ps)", ylabel="Temperature (K)", title="Temperature")

    axes[1].plot(t_ps, pes, alpha=0.3, lw=0.5)
    if w > 1:
        axes[1].plot(t_ps[w - 1:], np.convolve(pes, np.ones(w) / w, "valid"), "r", lw=1.5)
    axes[1].set(xlabel="Time (ps)", ylabel="PE (eV/atom)", title="Potential Energy")

    if densities and len(densities) > 0:
        t_rho = np.arange(len(densities)) * LOG_EVERY * 0.001
        axes[2].plot(t_rho, densities, alpha=0.3, lw=0.5)
        if w > 1:
            axes[2].plot(t_rho[w - 1:], np.convolve(densities, np.ones(w) / w, "valid"), "r", lw=1.5)
        converged, drift, mean_rho = check_convergence(densities)
        status = f"CONVERGED (rho={mean_rho:.4f})" if converged else f"drift={drift:.1f}%"
        axes[2].set(xlabel="Time (ps)", ylabel="Density (g/cm3)", title=f"Density, {status}")

    plt.suptitle(name, fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(box_dir(checkpoint_dir, name), "diagnostics.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Diagnostics: {out_path}")


def print_status(checkpoint_dir):
    from ase.io import read

    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory at {checkpoint_dir}")
        return

    dirs = sorted([d for d in os.listdir(checkpoint_dir)
                   if os.path.isdir(os.path.join(checkpoint_dir, d))])
    if not dirs:
        print("No boxes found.")
        return

    print(f"\n{'Box':<50s} {'Status':<20s} {'Density':<12s} {'Atoms'}")
    print("-" * 95)

    for d in dirs:
        dd = os.path.join(checkpoint_dir, d)
        final = os.path.join(dd, f"{d}.xyz")
        if os.path.exists(final):
            a = read(final)
            has_nan = np.any(np.isnan(a.get_positions()))
            if has_nan:
                print(f"{d:<50s} {'NaN!':<20s} {'N/A':<12s} {len(a)}")
            else:
                rho = sum(a.get_masses()) / a.get_volume() * 1.66054
                print(f"{d:<50s} {'DONE':<20s} {rho:.4f} g/cm3  {len(a)}")
        else:
            ckpts = sorted(glob.glob(os.path.join(dd, "ckpt_*.npz")))
            if ckpts:
                data = np.load(ckpts[-1], allow_pickle=True)
                phase = str(data["phase"])
                step = int(data["step"])
                rho = ""
                if "densities" in data and len(data["densities"]) > 0:
                    rho = f"{data['densities'][-1]:.4f} g/cm3"
                print(f"{d:<50s} {phase} step {step:<12} {rho}")
            else:
                print(f"{d:<50s} {'empty':<20s}")


def main():
    parser = argparse.ArgumentParser(
        description="Electrolyte box equilibration with checkpointing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", nargs="?",
                        help="Input .xyz file (FIRE-optimized or NVT-equilibrated)")
    parser.add_argument("--name",
                        help="Box name (default: from filename)")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
                        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", default="uma",
                        choices=["orbmol_v2", "uma"],
                        help="ML potential (default: uma)")
    parser.add_argument("--workflow", default="npt",
                        choices=["npt", "nvt", "nvt+npt"],
                        help="Equilibration workflow (default: npt)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", metavar="BOX_NAME",
                      help="Resume a previously interrupted run")
    mode.add_argument("--status", action="store_true",
                      help="Print status of all boxes and exit")

    parser.add_argument("--nvt-steps", type=int, default=NVT_STEPS,
                        help=f"NVT steps (default: {NVT_STEPS})")
    parser.add_argument("--npt-steps", type=int, default=NPT_STEPS,
                        help=f"NPT steps (default: {NPT_STEPS})")

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.status:
        print_status(args.checkpoint_dir)
        return

    if args.resume:
        name = args.resume
        atoms, step, phase, prev_t, prev_pe, prev_rho = load_checkpoint(
            args.checkpoint_dir, name)
        if atoms is None:
            print(f"No checkpoint found for '{name}' in {args.checkpoint_dir}")
            sys.exit(1)

        print(f"\nResuming: {name} (model: {args.model})")
        print(f"  Phase: {phase}, Step: {step}, Atoms: {len(atoms)}")

        all_temps, all_pes, all_densities = list(prev_t), list(prev_pe), list(prev_rho)

        if phase == "nvt" and step < args.nvt_steps:
            print(f"\nResuming NVT: {name} from step {step}")
            atoms, temps, pes = run_nvt(
                atoms, name, args.checkpoint_dir, args.model,
                n_steps=args.nvt_steps, start_step=step,
                prev_temps=prev_t, prev_pes=prev_pe, device=args.device)
            all_temps, all_pes = list(temps), list(pes)

            if args.workflow == "nvt+npt":
                print(f"\nNPT: {name}")
                atoms, temps, pes, densities = run_npt(
                    atoms, name, args.checkpoint_dir, args.model,
                    n_steps=args.npt_steps, device=args.device)
                all_temps.extend(temps); all_pes.extend(pes)
                all_densities = list(densities)

        elif phase == "nvt" and step >= args.nvt_steps:
            if args.workflow == "nvt+npt":
                print(f"\nNPT: {name}")
                atoms, temps, pes, densities = run_npt(
                    atoms, name, args.checkpoint_dir, args.model,
                    n_steps=args.npt_steps, device=args.device)
                all_temps.extend(temps); all_pes.extend(pes)
                all_densities = list(densities)
            else:
                print("NVT already complete.")

        elif phase == "npt" and step < args.npt_steps:
            print(f"\nResuming NPT: {name} from step {step}")
            atoms, temps, pes, densities = run_npt(
                atoms, name, args.checkpoint_dir, args.model,
                n_steps=args.npt_steps, start_step=step,
                prev_temps=prev_t, prev_pes=prev_pe,
                prev_densities=prev_rho, device=args.device)
            all_temps, all_pes = list(temps), list(pes)
            all_densities = list(densities)

        elif phase == "npt" and step >= args.npt_steps:
            print("Already fully equilibrated.")
            return

        plot_diagnostics(name, args.checkpoint_dir, all_temps, all_pes,
                         all_densities if all_densities else None)
        print(f"\nDone. Results in {box_dir(args.checkpoint_dir, name)}/")
        return

    if not args.input:
        parser.error("Provide an input .xyz file, --resume BOX_NAME, or --status")

    from ase.io import read, write

    name = args.name or os.path.basename(args.input).replace("_opt.xyz", "").replace(".xyz", "")

    print(f"\nBox:          {name}")
    print(f"Input:        {args.input}")
    print(f"Model:        {args.model}")
    print(f"Workflow:     {args.workflow}")
    print(f"Checkpoints:  {args.checkpoint_dir}/{name}/")
    print(f"Device:       {args.device}")

    atoms = read(args.input)
    print(f"Loaded: {len(atoms)} atoms, cell = {atoms.cell.lengths()}")

    all_temps, all_pes, all_densities = [], [], []

    if args.workflow in ("nvt", "nvt+npt"):
        print(f"\nNVT thermalization: {name}")
        atoms, temps, pes = run_nvt(
            atoms, name, args.checkpoint_dir, args.model,
            n_steps=args.nvt_steps, device=args.device)
        all_temps.extend(temps); all_pes.extend(pes)

        nvt_path = os.path.join(box_dir(args.checkpoint_dir, name), f"{name}_nvt.xyz")
        write(nvt_path, atoms, format="extxyz")
        print(f"  NVT structure: {nvt_path}")

    if args.workflow in ("npt", "nvt+npt"):
        print(f"\nNPT density equilibration: {name}")
        atoms, temps, pes, densities = run_npt(
            atoms, name, args.checkpoint_dir, args.model,
            n_steps=args.npt_steps, device=args.device)
        all_temps.extend(temps); all_pes.extend(pes)
        all_densities = list(densities)

    plot_diagnostics(name, args.checkpoint_dir, all_temps, all_pes,
                     all_densities if all_densities else None)

    d = box_dir(args.checkpoint_dir, name)
    print(f"\nDone. Results in {d}/")
    for f in sorted(os.listdir(d)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
