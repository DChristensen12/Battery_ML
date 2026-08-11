#!/usr/bin/env python3
"""Runs MD equilibration with an ML potential through ASE.

Covers NVT (Langevin), NPT (Nose-Hoover), and annealing (cyclic heating and
cooling). Defaults to orbmol_v2, which is trained on organic molecules and
battery electrolytes specifically.

Outputs:
    trajectory.traj  (ASE trajectory: positions, cell, velocities)
    md.log           (properties vs time: T, E_pot, E_kin, E_tot)
    final.xyz        (last frame)

Requirements:
    pip install ase torch orb-models    (or mace-torch)

Examples
--------
python run_md.py --project ./my-project -p nvt -T 300 -n 50000
python run_md.py --project ./my-project -p npt -T 300 -P 1.0 -n 100000
python run_md.py --project ./my-project -p anneal --t-low 300 --t-high 500 \
    --total-steps 200000 --num-cycles 5
"""

import argparse
import os
import sys

from utils import (
    ATM_TO_GPA, ATM_TO_EV_A3, DEFAULT_MODEL, DEFAULT_TIMESTEP_FS,
    DEFAULT_TRAJ_INTERVAL, DEFAULT_PROP_INTERVAL,
    ProjectLayout, get_calculator,
)


def clean_previous_outputs(output_dir: str):
    """Wipe leftovers from a previous run so two runs don't get mixed together."""
    removed = []
    for name in ("trajectory.traj", "md.log", "final.xyz"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)
            removed.append(name)
    if removed:
        print(f"  Cleaned: {', '.join(removed)}")


def setup_md(atoms, integrator, temperature, timestep_fs, friction,
             pressure_atm=None):
    """Builds the ASE dynamics object for whichever ensemble you asked for."""
    from ase import units
    from ase.md.langevin import Langevin

    if integrator == "nvt":
        dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature,
            friction=friction / units.fs,
        )
    elif integrator == "npt":
        from ase.md.npt import NPT
        pressure_eV_A3 = pressure_atm * ATM_TO_EV_A3
        bulk_mod_eV_A3 = 1.0 * 0.006242  # ~1 GPa for organic liquids
        dyn = NPT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature,
            externalstress=pressure_eV_A3,
            ttime=25 * units.fs,
            pfactor=(75 * units.fs) ** 2 * bulk_mod_eV_A3,
        )
    else:
        raise ValueError(f"Unknown integrator: {integrator}")

    return dyn


def attach_loggers(dyn, atoms, output_dir, traj_interval, prop_interval):
    from ase.io.trajectory import Trajectory
    from ase.md import MDLogger

    traj_path = os.path.join(output_dir, "trajectory.traj")
    log_path = os.path.join(output_dir, "md.log")

    traj = Trajectory(traj_path, "w", atoms)
    dyn.attach(traj.write, interval=traj_interval)

    logger = MDLogger(
        dyn, atoms, log_path,
        header=True, stress=False, peratom=False, mode="w",
    )
    dyn.attach(logger, interval=prop_interval)

    return traj


def run_nvt(args, atoms):
    dyn = setup_md(atoms, "nvt", args.temperature, args.timestep,
                   args.friction)
    traj = attach_loggers(dyn, atoms, args.output_dir,
                          args.traj_interval, args.prop_interval)

    print(f"\nRunning NVT: {args.temperature} K, {args.steps} steps...")
    dyn.run(args.steps)
    traj.close()


def run_npt(args, atoms):
    dyn = setup_md(atoms, "npt", args.temperature, args.timestep,
                   args.friction, pressure_atm=args.pressure)
    traj = attach_loggers(dyn, atoms, args.output_dir,
                          args.traj_interval, args.prop_interval)

    print(f"\nRunning NPT: {args.temperature} K, {args.pressure} atm, "
          f"{args.steps} steps...")
    dyn.run(args.steps)
    traj.close()


def run_anneal(args, atoms):
    """Heat up, cool back down, repeat for however many cycles you asked for.

    Temperature gets nudged every ramp_update steps, otherwise the ramp moves in
    visible jumps.
    """
    from ase import units
    from ase.io.trajectory import Trajectory
    from ase.md import MDLogger
    from ase.md.langevin import Langevin

    steps_per_segment = args.total_steps // (2 * args.num_cycles)
    ramp_update = max(1, min(100, steps_per_segment // 50))

    print(f"\nAnnealing: {args.t_low}->{args.t_high} K, "
          f"{args.num_cycles} cycles, {steps_per_segment} steps/segment")

    dyn = Langevin(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.t_low,
        friction=args.friction / units.fs,
    )

    traj_path = os.path.join(args.output_dir, "trajectory.traj")
    log_path = os.path.join(args.output_dir, "md.log")

    traj = Trajectory(traj_path, "w", atoms)
    dyn.attach(traj.write, interval=args.traj_interval)

    logger = MDLogger(dyn, atoms, log_path, header=True,
                      stress=False, peratom=False, mode="w")
    dyn.attach(logger, interval=args.prop_interval)

    for cycle in range(args.num_cycles):
        for phase, (t_start, t_end) in enumerate([
            (args.t_low, args.t_high),
            (args.t_high, args.t_low),
        ]):
            phase_name = "heating" if phase == 0 else "cooling"
            print(f"  Cycle {cycle+1}/{args.num_cycles}, {phase_name} "
                  f"({t_start}->{t_end} K)")

            steps_done = 0
            while steps_done < steps_per_segment:
                chunk = min(ramp_update, steps_per_segment - steps_done)
                frac = steps_done / max(1, steps_per_segment - 1)
                t_now = t_start + (t_end - t_start) * frac
                dyn.set_temperature(temperature_K=t_now)
                dyn.run(chunk)
                steps_done += chunk

    traj.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run MD equilibration with an ML potential (ASE).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project",
                        help="Project root, auto-derives input/output paths.")
    parser.add_argument("-i", "--input",
                        help="Input structure (PDB/XYZ with cell info).")
    parser.add_argument("-p", "--protocol", required=True,
                        choices=["nvt", "npt", "anneal"])
    parser.add_argument("--output-dir",
                        help="Output directory for trajectory + log.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"ML potential model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--device", default=None,
                        help="'cuda' or 'cpu' (auto-detected if omitted).")

    temp = parser.add_argument_group("temperature")
    temp.add_argument("-T", "--temperature", type=float, default=300.0,
                      help="Temperature in K (NVT/NPT, default: 300).")

    npt = parser.add_argument_group("NPT")
    npt.add_argument("-P", "--pressure", type=float, default=1.0,
                     help="Pressure in atm (NPT, default: 1.0).")

    ann = parser.add_argument_group("annealing")
    ann.add_argument("--t-low", type=float, default=300.0)
    ann.add_argument("--t-high", type=float, default=500.0)
    ann.add_argument("--total-steps", type=int, default=200000)
    ann.add_argument("--num-cycles", type=int, default=5)

    md = parser.add_argument_group("MD parameters")
    md.add_argument("-n", "--steps", type=int, default=50000,
                    help="Steps (NVT/NPT, default: 50000).")
    md.add_argument("-dt", "--timestep", type=float,
                    default=DEFAULT_TIMESTEP_FS,
                    help=f"Timestep in fs (default: {DEFAULT_TIMESTEP_FS}).")
    md.add_argument("--friction", type=float, default=0.01,
                    help="Langevin friction in 1/fs (default: 0.01).")
    md.add_argument("--traj-interval", type=int,
                    default=DEFAULT_TRAJ_INTERVAL)
    md.add_argument("--prop-interval", type=int,
                    default=DEFAULT_PROP_INTERVAL)

    parser.add_argument("--force", action="store_true",
                        help="Remove previous outputs before running.")

    args = parser.parse_args()

    if args.project:
        layout = ProjectLayout(args.project)
        args.input = args.input or layout.packed_pdb
        args.output_dir = args.output_dir or layout.equilibration_dir(args.protocol)

    if not args.input or not args.output_dir:
        parser.error("Provide --project, or both --input and --output-dir.")

    os.makedirs(args.output_dir, exist_ok=True)

    traj_path = os.path.join(args.output_dir, "trajectory.traj")
    if os.path.exists(traj_path) and not args.force:
        print(f"ERROR: {traj_path} already exists. Use --force to overwrite.")
        sys.exit(1)
    if args.force:
        clean_previous_outputs(args.output_dir)

    print(f"Protocol:  {args.protocol.upper()}")
    print(f"Input:     {args.input}")
    print(f"Output:    {args.output_dir}")
    print(f"Timestep:  {args.timestep} fs")

    from ase.io import read, write
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    print(f"\nLoading structure from {args.input}...")
    atoms = read(args.input)
    print(f"  {len(atoms)} atoms, cell = {atoms.cell.lengths()}")

    print("Loading ML potential...")
    calc = get_calculator(args.model, device=args.device)
    atoms.calc = calc

    t_init = args.temperature if args.protocol != "anneal" else args.t_low
    MaxwellBoltzmannDistribution(atoms, temperature_K=t_init)

    if args.protocol == "nvt":
        run_nvt(args, atoms)
    elif args.protocol == "npt":
        run_npt(args, atoms)
    elif args.protocol == "anneal":
        run_anneal(args, atoms)

    final_path = os.path.join(args.output_dir, "final.xyz")
    write(final_path, atoms)

    print(f"\nDone. Results in {args.output_dir}/")
    print("  trajectory.traj  (ASE trajectory)")
    print("  md.log           (property time series)")
    print("  final.xyz        (final structure)")


if __name__ == "__main__":
    main()
