#!/usr/bin/env python3
"""Unconstrained NVT MD on a La3+ droplet with FAIRChem UMA OMOL.

Langevin dynamics, no periodic boundaries, built for the Perlmutter A100s.

La3+ is [Xe]4f0, so it's closed shell with no unpaired electrons, and every
anion in this set is closed shell too. All five systems are singlets. The
omol head reads charge and spin straight off atoms.info, so those keys have
to survive the round trip through the xyz file or you're quietly running the
wrong system.

Usage:
  python run_uma_md.py La3+_F_droplet.xyz -o results/La3+_F/
  python run_uma_md.py La3+_F_droplet.xyz -o results/La3+_F/ --resume
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
from ase import units
from ase.constraints import FixCom
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# TF32 on the tensor cores, roughly 2x on matmul. The accuracy loss doesn't
# matter at MD resolution.
torch.set_float32_matmul_precision("high")


def get_uma_calculator(device="cuda"):
    """Load FAIRChem UMA-s-1.2 (OMol25) with turbo inference settings."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=device)

    # checkpointing trades speed for VRAM, which only ever pays off in training
    for attr in ("model", "inference_model"):
        if hasattr(predictor, attr):
            m = getattr(predictor, attr)
            if hasattr(m, "use_checkpoint"):
                m.use_checkpoint = False

    # fused kernels, if this model takes it. benchmark it, the edge count moves
    # every step and recompile churn can cost more than it saves
    try:
        if hasattr(predictor, "model"):
            predictor.model = torch.compile(predictor.model)
    except Exception:
        pass

    return FAIRChemCalculator(predictor, task_name="omol")


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    traj_path = os.path.join(args.output, "trajectory.xyz")
    log_path = os.path.join(args.output, "md.log")
    ckpt_path = os.path.join(args.output, "checkpoint.pkl")
    final_path = os.path.join(args.output, "final.xyz")

    atoms = read(args.input)
    atoms.pbc = False
    atoms.set_constraint(FixCom())

    charge = atoms.info.get("charge", 0)
    spin = atoms.info.get("spin", 1)

    print(f"System:  {os.path.basename(args.input)}")
    print(f"Atoms:   {len(atoms)}   Charge: {charge}   Spin: {spin}")
    print(f"Device:  {device}")
    print(f"Steps:   {args.steps:,} ({args.steps * args.dt / 1e6:.1f} ns)")
    print(f"Temp:    {args.temp} K   Friction: {args.friction}/fs")
    print(f"Output:  {args.output}")

    print("\nLoading UMA-s-1.2 (OMol25, turbo)...", flush=True)
    calc = get_uma_calculator(device)
    atoms.calc = calc

    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        atoms.positions = ckpt["positions"]
        atoms.set_velocities(ckpt["velocities"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step:,} ({start_step * args.dt / 1e6:.3f} ns)")
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp)
        print("Initialized Maxwell-Boltzmann velocities")
        if os.path.exists(traj_path):
            os.remove(traj_path)

    dyn = Langevin(
        atoms,
        timestep=args.dt * units.fs,
        temperature_K=args.temp,
        friction=args.friction / units.fs,
    )

    mode = "a" if start_step > 0 else "w"
    log_fh = open(log_path, mode)
    if start_step == 0:
        log_fh.write("# step  time_ps  T_K  PE_eV  KE_eV\n")

    ndof = 3 * len(atoms) - 3  # FixCom removes 3 translational DOF

    print(f"\nRunning: step {start_step:,} -> {args.steps:,}\n", flush=True)
    t0 = time.time()
    step = start_step

    while step < args.steps:
        dyn.run(1)
        step += 1

        if step % args.log_interval == 0:
            ke = atoms.get_kinetic_energy()
            pe = atoms.get_potential_energy()
            temp = 2 * ke / (ndof * units.kB)
            t_ps = step * args.dt / 1000
            log_fh.write(f"{step}  {t_ps:.3f}  {temp:.2f}  {pe:.6f}  {ke:.6f}\n")
            log_fh.flush()

        if step % args.traj_interval == 0:
            write(traj_path, atoms, append=True)

        if step % args.snap_interval == 0:
            write(os.path.join(args.output, f"snap_{step}.xyz"), atoms)

        if step % args.ckpt_interval == 0:
            tmp = ckpt_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({
                    "positions": atoms.positions.copy(),
                    "velocities": atoms.get_velocities().copy(),
                    "step": step,
                }, f)
            os.replace(tmp, ckpt_path)

        if step % 10_000 == 0:
            elapsed = time.time() - t0
            speed = (step - start_step) / elapsed if elapsed > 0 else 0
            ke = atoms.get_kinetic_energy()
            temp = 2 * ke / (ndof * units.kB)
            t_ns = step * args.dt / 1e6
            eta_h = (args.steps - step) / speed / 3600 if speed > 0 else float("inf")
            print(
                f"  step={step:>10,}  t={t_ns:.3f}ns  T={temp:.0f}K  "
                f"{speed:.1f}st/s  ETA={eta_h:.1f}h",
                flush=True,
            )

    write(final_path, atoms)
    log_fh.close()

    elapsed = time.time() - t0
    print(f"\nDone: {step:,} steps ({step * args.dt / 1e6:.1f} ns) in {elapsed / 3600:.1f}h")
    print(f"Final:      {final_path}")
    print(f"Trajectory: {traj_path}")
    print(f"Log:        {log_path}")


def main():
    p = argparse.ArgumentParser(description="UMA OMOL unconstrained MD")
    p.add_argument("input", help="input .xyz file")
    p.add_argument("-o", "--output", required=True, help="output directory")
    p.add_argument("--steps", type=int, default=10_000_000,
                    help="total MD steps (default: 10M = 10 ns)")
    p.add_argument("--dt", type=float, default=1.0, help="timestep in fs")
    p.add_argument("--temp", type=float, default=300, help="temperature in K")
    p.add_argument("--friction", type=float, default=0.01,
                    help="Langevin friction in 1/fs (default: 0.01)")
    p.add_argument("--log-interval", type=int, default=1000,
                    help="log every N steps (default: 1000 = 1 ps)")
    p.add_argument("--traj-interval", type=int, default=1000,
                    help="save trajectory every N steps (default: 1000 = 1 ps)")
    p.add_argument("--ckpt-interval", type=int, default=50_000,
                    help="checkpoint every N steps (default: 50000 = 50 ps)")
    p.add_argument("--snap-interval", type=int, default=500_000,
                    help="snapshot every N steps (default: 500000 = 500 ps)")
    p.add_argument("--resume", action="store_true",
                    help="resume from last checkpoint")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
