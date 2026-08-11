#!/usr/bin/env python3
"""Plots density and temperature vs time from an equilibration run.

Reads md.log (the property time series) and trajectory.traj (cell data) from
run_md.py and spits out diagnostic plots.

Requirements:
    pip install ase numpy matplotlib

Examples
--------
python analyze_trajectory.py --project ./my-project --protocol npt
python analyze_trajectory.py --log npt/md.log --traj npt/trajectory.traj \
    --pdb packed/system.pdb -o analysis/
"""

import argparse
import os
import re

import numpy as np

from utils import (
    parse_pdb_elements, total_mass_grams, ProjectLayout,
    DEFAULT_PROP_INTERVAL, DEFAULT_TRAJ_INTERVAL, DEFAULT_TIMESTEP_FS,
)

# fractal design system colors (I do like colors a lot)
PINK_300 = "#DC9ED3"
BLUE_300 = "#9FCEDB"
ORANGE_300 = "#DBB397"
GREY_400 = "#929295"
GREY_500 = "#6F6F72"
GREY_900 = "#1C1C1E"
BG_CANVAS = "#FEFEFE"
GRID_COLOR = "#E6E6E7"


def parse_md_log(log_path: str) -> dict[str, np.ndarray]:
    """Reads an ASE MDLogger file into arrays keyed by column name (Time[ps], Etot[eV], T[K], and so on)."""
    with open(log_path) as f:
        lines = [l for l in f if not l.startswith("#")]

    if not lines:
        raise ValueError(f"Empty log file: {log_path}")

    header = lines[0].split()
    data_lines = lines[1:]

    cols = {name: [] for name in header}
    for line in data_lines:
        vals = line.split()
        if len(vals) != len(header):
            continue
        for name, val in zip(header, vals):
            cols[name].append(float(val))

    return {name: np.array(vals) for name, vals in cols.items()}


def load_density_from_traj(traj_path: str, mass_g: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell volume per frame, converted to density in g/cm3."""
    from ase.io.trajectory import Trajectory

    traj = Trajectory(traj_path, "r")
    volumes = []
    for atoms in traj:
        volumes.append(atoms.get_volume())
    traj.close()

    volumes = np.array(volumes)
    densities = mass_g / (volumes * 1e-24)  # A^3 to cm^3
    return np.arange(len(volumes)), densities


def running_average(data: np.ndarray, window: int) -> np.ndarray:
    """Moving average by convolution. Returns the data untouched if the window is wider than it."""
    if window <= 1 or len(data) < window:
        return data.copy()
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def setup_matplotlib():
    """Agg backend so this works headless, plus the color scheme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (8, 4.5), "figure.dpi": 150,
        "figure.facecolor": BG_CANVAS, "axes.facecolor": BG_CANVAS,
        "axes.edgecolor": GREY_400, "axes.linewidth": 0.8,
        "axes.labelsize": "medium", "axes.labelweight": "500",
        "axes.labelcolor": GREY_500, "axes.titlesize": "large",
        "axes.titleweight": "600", "axes.titlecolor": GREY_900,
        "axes.grid": True, "axes.spines.right": False, "axes.spines.top": False,
        "grid.color": GRID_COLOR, "grid.linewidth": 0.6, "grid.alpha": 0.7,
        "xtick.color": GREY_400, "xtick.labelcolor": GREY_500,
        "ytick.color": GREY_400, "ytick.labelcolor": GREY_500,
        "lines.linewidth": 2.0, "savefig.dpi": 200,
        "savefig.facecolor": BG_CANVAS, "savefig.bbox": "tight",
    })
    return plt


def plot_series(plt, time, values, color, ylabel, title, window, output_path):
    """One time series, instantaneous plus running average."""
    fig, ax = plt.subplots()
    ax.plot(time, values, color=color, linewidth=1.0, alpha=0.5,
            label="Instantaneous")
    if len(values) > window:
        avg = running_average(values, window)
        ax.plot(time, avg, color=color, linewidth=2.0,
                label=f"Running avg ({window} frames)")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              frameon=True, facecolor="#FAFAFA", edgecolor=GRID_COLOR,
              labelcolor=GREY_500)
    fig.savefig(output_path, pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_stats(name: str, data: np.ndarray, unit: str):
    """Mean/std over the whole run and over the last quarter, which is the part that
    should look equilibrated. Drift is a straight line fit, only if there's enough data."""
    last_quarter = data[3 * len(data) // 4:]
    print(f"\n  {name}:")
    print(f"    Overall:  mean={np.mean(data):.4f}, std={np.std(data):.4f} {unit}")
    print(f"    Last 25%: mean={np.mean(last_quarter):.4f}, "
          f"std={np.std(last_quarter):.4f} {unit}")
    if len(data) > 10:
        drift = np.polyfit(np.arange(len(data)), data, 1)[0]
        print(f"    Drift:    {drift:.6e} {unit}/frame")


def main():
    parser = argparse.ArgumentParser(
        description="Plot equilibration diagnostics (temperature, density, energy vs time).",
    )
    parser.add_argument("--project",
                        help="Project root, auto-derives all paths.")
    parser.add_argument("--protocol", default="npt",
                        choices=["nvt", "npt", "anneal"],
                        help="Which equilibration to analyze (with --project).")
    parser.add_argument("--log", help="Path to md.log from run_md.py.")
    parser.add_argument("--traj", help="Path to trajectory.traj.")
    parser.add_argument("--pdb", help="PDB for atom masses (density calc).")
    parser.add_argument("-o", "--output-dir", help="Directory for plots.")
    parser.add_argument("-w", "--window", type=int, default=200,
                        help="Running average window (default: 200).")

    args = parser.parse_args()

    if args.project:
        layout = ProjectLayout(args.project)
        args.log = args.log or layout.md_log(args.protocol)
        args.traj = args.traj or layout.trajectory(args.protocol)
        args.pdb = args.pdb or layout.packed_pdb
        args.output_dir = args.output_dir or layout.analysis

    if not args.log or not args.output_dir:
        parser.error("Provide --project, or at least --log and --output-dir.")

    os.makedirs(args.output_dir, exist_ok=True)
    plt = setup_matplotlib()

    # temp and energy come straight from md.log
    print(f"Reading {args.log}...")
    log = parse_md_log(args.log)

    time_key = [k for k in log if "time" in k.lower()]
    time_ps = log[time_key[0]] if time_key else np.arange(len(next(iter(log.values()))))

    temp_key = [k for k in log if k.startswith("T[") or k == "T"]
    if temp_key:
        temp = log[temp_key[0]]
        print_stats("Temperature", temp, "K")
        plot_series(plt, time_ps, temp, PINK_300, "Temperature (K)",
                    "Temperature vs Time", args.window,
                    os.path.join(args.output_dir, "temperature_vs_time.png"))

    epot_key = [k for k in log if "pot" in k.lower()]
    if epot_key:
        epot = log[epot_key[0]]
        print_stats("Potential Energy", epot, "eV")
        plot_series(plt, time_ps, epot, ORANGE_300, "Potential Energy (eV)",
                    "Potential Energy vs Time", args.window,
                    os.path.join(args.output_dir, "energy_vs_time.png"))

    # density needs the trajectory, since it comes from cell volume
    if args.traj and args.pdb:
        print(f"\nReading trajectory for density: {args.traj}")
        elements = parse_pdb_elements(args.pdb)
        mass_g = total_mass_grams(elements)
        print(f"  {len(elements)} atoms, total mass = {mass_g:.4e} g")

        frame_idx, density = load_density_from_traj(args.traj, mass_g)

        time_traj = frame_idx * DEFAULT_TIMESTEP_FS * DEFAULT_TRAJ_INTERVAL * 1e-3
        print_stats("Density", density, "g/cm\u00B3")
        plot_series(plt, time_traj, density, BLUE_300, "Density (g/cm\u00B3)",
                    "Density vs Time", min(args.window, len(density) // 2),
                    os.path.join(args.output_dir, "density_vs_time.png"))
    elif args.traj and not args.pdb:
        print("\n  Skipping density plot, provide --pdb for atom masses.")

    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
