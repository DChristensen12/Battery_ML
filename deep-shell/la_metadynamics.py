#!/usr/bin/env python3
"""Well-tempered metadynamics of La3+ coordination with OH- and F- in a water droplet.

Langevin NVT with MACE-polar forces, biased along the La-anion coordination
number. Checkpoints go to ~/la_metadynamics/<system>/ so a killed run picks up
where it left off. Only the latest checkpoint per system is kept.

Outputs per system:
    ckpt_<step>.xyz     latest structure
    metad_state.npz     hills, colvar, and the metad parameters
    HILLS.txt           step, cn, weight_eV
    COLVAR.txt          step, cn, bias_eV, energy_eV, temp_K

Examples
--------
# Fresh run from an xyz
python la_metadynamics.py La_OH_droplet.xyz --system oh --steps 200000

# Resume an interrupted run
python la_metadynamics.py --resume oh

# Status of all systems
python la_metadynamics.py --status

# Reconstruct the FES and plot
python la_metadynamics.py --analyze

# Build a droplet instead of supplying one
python la_metadynamics.py --build oh
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.calculator import Calculator, all_changes
import ase.units as units

KJ_TO_EV = 1.0 / 96.485
EV_TO_KJ = 96.485

DEFAULT_CKPT_DIR = Path.home() / "la_metadynamics"

# r0 is the switching function cutoff, taken from the first minimum of the La-anion RDF
SYSTEMS = {
    "oh": {"label": "La3+ + 3 OH-", "r0": 3.5, "element": "O", "n_anions": 3},
    "f":  {"label": "La3+ + 3 F-",  "r0": 3.2, "element": "F", "n_anions": 3},
}


def smooth_cn_and_grad(positions, la_idx, anion_indices, r0, n=6, m=12):
    """CN = sum_i (1 - (r/r0)^n) / (1 - (r/r0)^m), and its gradient on every atom."""
    n_atoms = len(positions)
    la_pos = positions[la_idx]
    cn = 0.0
    grad = np.zeros((n_atoms, 3))

    for i in anion_indices:
        r_vec = positions[i] - la_pos
        r = np.linalg.norm(r_vec)
        if r < 1e-10:
            continue
        r_hat = r_vec / r
        u = r / r0

        u_n = u ** n
        u_m = u ** m
        # the switching function is 0/0 exactly at r = r0, clamp rather than special case it
        denom = max(1.0 - u_m, 1e-12)

        cn += (1.0 - u_n) / denom

        dcn_dr = (1.0 / r0) * (
            -n * u ** (n - 1) * (1.0 - u_m)
            + m * u ** (m - 1) * (1.0 - u_n)
        ) / (denom ** 2)

        grad[i] += dcn_dr * r_hat
        grad[la_idx] -= dcn_dr * r_hat

    return cn, grad


class WellTemperedMetadynamics:
    def __init__(self, sigma=0.15, height_kj=2.0, pace=500,
                 bias_factor=15, temperature=300):
        self.sigma = sigma
        self.w0 = height_kj * KJ_TO_EV
        self.pace = pace
        self.gamma = bias_factor
        self.kBT = 8.617e-5 * temperature
        self.hills = []   # [(cn, weight_eV), ...]
        self.colvar = []  # [(step, cn, bias_eV, energy_eV, temp_K), ...]
        self.step = 0

    def bias_potential(self, cn):
        v = 0.0
        for cn_k, w_k in self.hills:
            v += w_k * np.exp(-(cn - cn_k) ** 2 / (2 * self.sigma ** 2))
        return v

    def bias_gradient(self, cn):
        dv = 0.0
        for cn_k, w_k in self.hills:
            g = np.exp(-(cn - cn_k) ** 2 / (2 * self.sigma ** 2))
            dv += w_k * g * (-(cn - cn_k) / self.sigma ** 2)
        return dv

    def deposit_hill(self, cn):
        # well-tempered part: hills shrink where the bias is already deep, gamma sets how fast
        v_current = self.bias_potential(cn)
        w = self.w0 * np.exp(-v_current / (self.kBT * (self.gamma - 1)))
        self.hills.append((cn, w))
        return w

    def record_colvar(self, step, cn, energy, temperature):
        self.colvar.append((step, cn, self.bias_potential(cn), energy, temperature))

    def save(self, path):
        path = Path(path)
        hills_arr = np.array(self.hills) if self.hills else np.empty((0, 2))
        colvar_arr = np.array(self.colvar) if self.colvar else np.empty((0, 5))

        np.savez(
            path / "metad_state.npz",
            sigma=self.sigma, w0_eV=self.w0, pace=self.pace,
            gamma=self.gamma, kBT_eV=self.kBT, step=self.step,
            hills=hills_arr, colvar=colvar_arr,
        )

        if self.hills:
            steps = np.arange(1, len(hills_arr) + 1) * self.pace
            np.savetxt(
                path / "HILLS.txt",
                np.column_stack([steps, hills_arr]),
                header="step cn weight_eV", fmt="%.0f %.6f %.8f",
            )
        if self.colvar:
            np.savetxt(
                path / "COLVAR.txt", colvar_arr,
                header="step cn bias_eV energy_eV temp_K",
                fmt="%.0f %.6f %.6f %.4f %.1f",
            )

    def load(self, path):
        path = Path(path)
        data = np.load(path / "metad_state.npz", allow_pickle=True)
        self.sigma = float(data["sigma"])
        self.w0 = float(data["w0_eV"])
        self.pace = int(data["pace"])
        self.gamma = float(data["gamma"])
        self.kBT = float(data["kBT_eV"])
        self.step = int(data["step"])
        h = data["hills"]
        self.hills = [(float(r[0]), float(r[1])) for r in h] if len(h) else []
        c = data["colvar"]
        self.colvar = [tuple(r) for r in c] if len(c) else []


class BiasedCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, metad, la_idx, anion_indices, r0):
        super().__init__()
        self.base_calc = base_calc
        self.metad = metad
        self.la_idx = la_idx
        self.anion_indices = anion_indices
        self.r0 = r0
        self.last_cn = 0.0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy", "forces"]
        super().calculate(atoms, properties, system_changes)

        self.base_calc.calculate(atoms, properties, system_changes)
        base_e = self.base_calc.results["energy"]
        base_f = self.base_calc.results["forces"].copy()

        cn, dcn = smooth_cn_and_grad(
            atoms.positions, self.la_idx, self.anion_indices, self.r0,
        )
        v_bias = self.metad.bias_potential(cn)
        dv_dcn = self.metad.bias_gradient(cn)

        self.results["energy"] = base_e + v_bias
        # chain rule, -dV/dx = -(dV/dCN)(dCN/dx)
        self.results["forces"] = base_f - dv_dcn * dcn
        self.last_cn = cn


def identify_atoms(atoms, system_type):
    symbols = atoms.get_chemical_symbols()
    cfg = SYSTEMS[system_type]

    la_indices = [i for i, s in enumerate(symbols) if s == "La"]
    if not la_indices:
        raise ValueError("No La atom found in structure")
    la_idx = la_indices[0]

    # nearest n by distance, which for oh will grab a water O if an anion has already
    # wandered off. only run this on a fresh droplet, not on a resumed frame
    la_pos = atoms.positions[la_idx]
    candidates = sorted(
        [(i, np.linalg.norm(atoms.positions[i] - la_pos))
         for i, s in enumerate(symbols) if s == cfg["element"]],
        key=lambda x: x[1],
    )
    anion_indices = [idx for idx, _ in candidates[: cfg["n_anions"]]]
    return la_idx, anion_indices


def get_calculator(model_size="medium", device=None):
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from mace.calculators import mace_polar
    size_map = {"small": "polar-1-s", "medium": "polar-1-m", "large": "polar-1-l"}
    model_name = size_map.get(model_size, model_size)
    calc = mace_polar(model=model_name, device=device, default_dtype="float64")
    print(f"Loaded MACE-polar ({model_name}) on {device}")
    return calc


def save_checkpoint(atoms, metad, system_type, ckpt_dir, step):
    sys_dir = Path(ckpt_dir) / system_type
    sys_dir.mkdir(parents=True, exist_ok=True)

    for old in sys_dir.glob("ckpt_*.xyz"):
        old.unlink()

    write(sys_dir / f"ckpt_{step}.xyz", atoms)
    metad.step = step
    metad.save(sys_dir)
    return sys_dir / f"ckpt_{step}.xyz"


def load_checkpoint(system_type, ckpt_dir):
    sys_dir = Path(ckpt_dir) / system_type
    if not sys_dir.exists():
        return None

    ckpts = sorted(sys_dir.glob("ckpt_*.xyz"))
    if not ckpts:
        return None

    latest = ckpts[-1]
    step = int(latest.stem.split("_")[1])
    atoms = read(latest)

    metad = WellTemperedMetadynamics()
    metad.load(sys_dir)

    print(f"Resumed {system_type} from step {step} ({step / 1000:.1f} ps)")
    if metad.hills:
        cn_vals = [h[0] for h in metad.hills]
        print(f"  {len(metad.hills)} hills, CN range [{min(cn_vals):.2f}, {max(cn_vals):.2f}]")
    return atoms, metad, step


def get_status(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        print("No checkpoint directory found.")
        return

    print(f"\n{'System':<8} {'Status':<22} {'Hills':<8} {'CN Range':<18} {'Last CN':<8}")

    for sys_type in SYSTEMS:
        sys_dir = ckpt_dir / sys_type
        if not sys_dir.exists():
            print(f"{sys_type:<8} {'not started':<22}")
            continue
        ckpts = sorted(sys_dir.glob("ckpt_*.xyz"))
        if not ckpts:
            print(f"{sys_type:<8} {'empty':<22}")
            continue

        step = int(ckpts[-1].stem.split("_")[1])
        metad = WellTemperedMetadynamics()
        try:
            metad.load(sys_dir)
        except (FileNotFoundError, KeyError, ValueError):
            print(f"{sys_type:<8} {'no readable state':<22}")
            continue

        n = len(metad.hills)
        if n:
            cn_vals = [h[0] for h in metad.hills]
            cn_range = f"[{min(cn_vals):.2f}, {max(cn_vals):.2f}]"
            last = f"{cn_vals[-1]:.2f}"
        else:
            cn_range = last = "-"

        status = f"step {step} ({step / 1000:.0f} ps)"
        print(f"{sys_type:<8} {status:<22} {n:<8} {cn_range:<18} {last:<8}")


def run_metadynamics(
    atoms, system_type, total_steps, model_size="medium", device=None,
    ckpt_dir=DEFAULT_CKPT_DIR, ckpt_interval=5000,
    start_step=0, metad=None,
    dt=1.0, temperature=300, friction=0.01,
    sigma=0.15, height_kj=2.0, pace=500, bias_factor=15,
):
    cfg = SYSTEMS[system_type]
    la_idx, anion_indices = identify_atoms(atoms, system_type)

    print(f"\nSystem: {cfg['label']}")
    print(f"  La index {la_idx}, anion indices {anion_indices}")
    print(f"  CN cutoff r0 = {cfg['r0']} A, steps {start_step} -> {total_steps}")

    base_calc = get_calculator(model_size, device)

    if metad is None:
        metad = WellTemperedMetadynamics(
            sigma=sigma, height_kj=height_kj, pace=pace,
            bias_factor=bias_factor, temperature=temperature,
        )

    biased = BiasedCalculator(base_calc, metad, la_idx, anion_indices, cfg["r0"])
    atoms.calc = biased

    if start_step == 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    dyn = Langevin(
        atoms, timestep=dt * units.fs,
        temperature_K=temperature, friction=friction / units.fs,
    )

    t0 = time.time()
    step = start_step
    log_every = 1000

    # one step at a time so last_cn is current when a hill lands
    while step < total_steps:
        dyn.run(1)
        step += 1
        metad.step = step
        cn = biased.last_cn

        if step % pace == 0:
            metad.deposit_hill(cn)
            # energy logged here is biased, subtract COLVAR bias_eV for the plain PE
            metad.record_colvar(step, cn, atoms.get_potential_energy(),
                                atoms.get_temperature())

        if step % log_every == 0:
            elapsed = time.time() - t0
            rate = (step - start_step) / elapsed if elapsed > 0 else 0
            eta_h = (total_steps - step) / rate / 3600 if rate > 0 else 0
            print(f"  step {step:>7d}/{total_steps}  CN={cn:.3f}  "
                  f"hills={len(metad.hills)}  {rate:.1f} st/s  ETA {eta_h:.1f}h")

        if step % ckpt_interval == 0:
            save_checkpoint(atoms, metad, system_type, ckpt_dir, step)
            converged, max_diff = check_convergence(metad)
            if converged:
                print(f"\n  *** FES converged (max delta = {max_diff:.2f} kJ/mol "
                      f"between 80% and 100% of hills) ***")
                break
            elif max_diff < float("inf"):
                print(f"  convergence check: max delta = {max_diff:.1f} kJ/mol")

    save_checkpoint(atoms, metad, system_type, ckpt_dir, step)
    elapsed = time.time() - t0
    print(f"\nDone: {step / 1000:.0f} ps in {elapsed / 3600:.1f} h, "
          f"{len(metad.hills)} hills deposited")
    return metad


def reconstruct_fes(hills, sigma, cn_grid, bias_factor):
    fes = np.zeros_like(cn_grid)
    for cn_k, w_k in hills:
        fes += w_k * np.exp(-(cn_grid - cn_k) ** 2 / (2 * sigma ** 2))
    # well-tempered bias only converges to -(gamma-1)/gamma of F, so scale it back up
    fes *= -(bias_factor / (bias_factor - 1))
    fes -= fes.min()
    return fes * EV_TO_KJ


def check_convergence(metad, tol_kj=0.5, min_hills=40):
    """Compare the FES from the first 80% of hills to the full set.

    Returns (converged, max_diff_kj).  Needs at least min_hills so the
    80/100 split is meaningful — before that it always returns False.
    """
    n = len(metad.hills)
    if n < min_hills:
        return False, float("inf")
    cn_grid = np.linspace(-0.2, 3.5, 300)
    n80 = int(n * 0.8)
    fes_80 = reconstruct_fes(metad.hills[:n80], metad.sigma, cn_grid, metad.gamma)
    fes_all = reconstruct_fes(metad.hills, metad.sigma, cn_grid, metad.gamma)
    max_diff = np.max(np.abs(fes_all - fes_80))
    return max_diff < tol_kj, max_diff


def load_results(ckpt_dir):
    results = {}
    for sys_type in SYSTEMS:
        sys_dir = Path(ckpt_dir) / sys_type
        if not sys_dir.exists():
            continue
        metad = WellTemperedMetadynamics()
        try:
            metad.load(sys_dir)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if metad.hills:
            results[sys_type] = metad
    return results


def analyze_and_plot(ckpt_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_dir = Path(ckpt_dir)
    results = load_results(ckpt_dir)
    if not results:
        print("No completed runs to analyze.")
        return

    cn_grid = np.linspace(-0.2, 3.5, 600)
    colors = {"oh": "#5CC2E1", "f": "#CB62BB"}
    labels = {st: cfg["label"] for st, cfg in SYSTEMS.items()}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for st, metad in results.items():
        fes = reconstruct_fes(metad.hills, metad.sigma, cn_grid, metad.gamma)
        n_ps = len(metad.colvar) * metad.pace / 1000
        ax.plot(cn_grid, fes, color=colors[st], lw=2.2,
                label=f"{labels[st]} ({n_ps:.0f} ps)")
    ax.set_xlabel("Coordination Number (La-anion)")
    ax.set_ylabel("F(CN) (kJ/mol)")
    ax.set_title("Free Energy Surface: La3+ OH- vs F- Coordination")
    ax.set_xlim(-0.1, 3.2)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.savefig(ckpt_dir / "fes_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved {ckpt_dir / 'fes_comparison.png'}")
    plt.close()

    n_sys = len(results)
    fig, axes = plt.subplots(n_sys, 1, figsize=(7, 3 * n_sys), squeeze=False)
    for ax_row, (st, metad) in zip(axes, results.items()):
        ax = ax_row[0]
        colvar = np.array(metad.colvar)
        ax.plot(colvar[:, 0] / 1000, colvar[:, 1], color=colors[st], lw=0.8, alpha=0.8)
        ax.set_ylabel(f"CN (La-{SYSTEMS[st]['element']})")
        ax.set_title(f"{labels[st]} ({colvar[-1, 0] / 1000:.0f} ps)")
        ax.set_ylim(-0.1, 3.2)
        for y in [1, 2, 3]:
            ax.axhline(y=y, color="#E6E6E7", ls="--", lw=0.6)
    axes[-1][0].set_xlabel("Time (ps)")
    fig.savefig(ckpt_dir / "cn_timeseries.png", dpi=200, bbox_inches="tight")
    print(f"Saved {ckpt_dir / 'cn_timeseries.png'}")
    plt.close()

    # thirds of the hill list, if the last two lie on top of each other the FES has converged
    fig, axes = plt.subplots(1, n_sys, figsize=(5 * n_sys, 4), sharey=True, squeeze=False)
    for ax_col, (st, metad) in zip(axes[0], results.items()):
        for frac, alpha, ls in [(0.33, 0.4, "--"), (0.66, 0.65, "-."), (1.0, 1.0, "-")]:
            k = max(1, int(len(metad.hills) * frac))
            fes = reconstruct_fes(metad.hills[:k], metad.sigma, cn_grid, metad.gamma)
            ps = int(k * metad.pace / 1000)
            ax_col.plot(cn_grid, fes, color=colors[st], lw=1.6,
                        alpha=alpha, ls=ls, label=f"{ps} ps")
        ax_col.set_title(f"{labels[st]} convergence")
        ax_col.set_xlabel("CN")
        ax_col.legend(frameon=False, fontsize=9)
    axes[0][0].set_ylabel("F(CN) (kJ/mol)")
    fig.savefig(ckpt_dir / "fes_convergence.png", dpi=200, bbox_inches="tight")
    print(f"Saved {ckpt_dir / 'fes_convergence.png'}")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    for st, metad in results.items():
        hills_arr = np.array(metad.hills)
        steps_ps = np.arange(1, len(hills_arr) + 1) * metad.pace / 1000
        ax.scatter(steps_ps, hills_arr[:, 1] * EV_TO_KJ,
                   s=12, color=colors[st], alpha=0.7, label=labels[st])
    ax.axhline(y=2.0, color="#929295", ls="--", lw=0.8, label="w0 = 2.0 kJ/mol")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Hill height (kJ/mol)")
    ax.set_title("(a) Well-tempered hill height decay")
    ax.set_ylim(0, 2.2)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for st, metad in results.items():
        colvar = np.array(metad.colvar)
        ax.hist(colvar[:, 1], bins=30, range=(0, 3.2), alpha=0.55,
                color=colors[st], label=labels[st], density=True)
    ax.set_xlabel("Coordination Number")
    ax.set_ylabel("Probability density")
    ax.set_title("(b) CN exploration (biased)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for st, metad in results.items():
        fes = reconstruct_fes(metad.hills, metad.sigma, cn_grid, metad.gamma)
        n_ps = len(metad.colvar) * metad.pace / 1000
        ax.plot(cn_grid, fes, color=colors[st], lw=2.2,
                label=f"{labels[st]} ({n_ps:.0f} ps)")
    if "oh" in results:
        fes_oh = reconstruct_fes(results["oh"].hills, results["oh"].sigma, cn_grid, results["oh"].gamma)
        ax.annotate("La(OH)3 stable\nCN ~ 3", xy=(2.7, fes_oh[np.argmin(np.abs(cn_grid - 2.7))]),
                    xytext=(2.8, max(fes_oh) * 0.7), fontsize=7.5, color="#6F6F72",
                    arrowprops=dict(arrowstyle="-", color="#929295", lw=0.7))
    if "f" in results:
        fes_f = reconstruct_fes(results["f"].hills, results["f"].sigma, cn_grid, results["f"].gamma)
        ax.annotate("F- dissociates\nin water", xy=(0.9, fes_f[np.argmin(np.abs(cn_grid - 0.9))]),
                    xytext=(0.2, max(fes_f) * 0.6), fontsize=7.5, color="#6F6F72",
                    arrowprops=dict(arrowstyle="-", color="#929295", lw=0.7))
    ax.set_xlabel("Coordination Number (La-anion)")
    ax.set_ylabel("F(CN) (kJ/mol)")
    ax.set_title("(c) Free energy surface comparison")
    ax.set_xlim(-0.1, 3.3)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for st, metad in results.items():
        colvar = np.array(metad.colvar)
        ax.plot(colvar[:, 0] / 1000, colvar[:, 2] * EV_TO_KJ,
                color=colors[st], lw=1.2, label=labels[st])
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("V_bias at current CN (kJ/mol)")
    ax.set_title("(d) Bias potential growth")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(ckpt_dir / "fes_validation_dashboard.png", dpi=200, bbox_inches="tight")
    print(f"Saved {ckpt_dir / 'fes_validation_dashboard.png'}")
    plt.close()

    for st, metad in results.items():
        fes = reconstruct_fes(metad.hills, metad.sigma, cn_grid, metad.gamma)
        colvar = np.array(metad.colvar)
        print(f"\n{labels[st]}")
        print(f"  {colvar[-1, 0] / 1000:.1f} ps, {len(metad.hills)} hills")
        print(f"  CN explored [{colvar[:, 1].min():.2f}, {colvar[:, 1].max():.2f}]")
        print(f"  FES minimum at CN = {cn_grid[np.argmin(fes)]:.2f}")
        # bound and free reference points differ, oh should still be intact at CN 3
        if st == "oh":
            dF = fes[np.argmin(np.abs(cn_grid - 1.5))] - fes[np.argmin(np.abs(cn_grid - 2.7))]
        else:
            dF = fes[np.argmin(np.abs(cn_grid - 0.5))] - fes[np.argmin(np.abs(cn_grid - 2.5))]
        print(f"  dF(bound->free) = {dF:+.1f} kJ/mol")


def build_droplet(system_type, n_waters=128, output_path=None):
    from ase import Atoms

    cfg = SYSTEMS[system_type]
    positions = [[0.0, 0.0, 0.0]]
    symbols = ["La"]

    r_coord = 2.45
    anion_dirs = [
        [1, 0, 0],
        [-0.5, 0.866, 0],
        [-0.5, -0.866, 0],
    ]

    if system_type == "oh":
        for d in anion_dirs:
            o_pos = [r_coord * x for x in d]
            positions.append(o_pos)
            symbols.append("O")
            h_pos = [(r_coord + 0.96) * x for x in d]
            positions.append(h_pos)
            symbols.append("H")
    else:
        for d in anion_dirs:
            positions.append([r_coord * x for x in d])
            symbols.append("F")

    rng = np.random.default_rng(42)
    r_sphere = 10.0
    r_min_inner = 3.5   # keeps solvent out of the first shell so the run starts near CN 3
    r_min_pair = 2.2

    # rejection sampling. sloppy but the droplet only needs to live through equilibration
    placed = 0
    for _ in range(n_waters * 500):
        if placed >= n_waters:
            break
        r = r_sphere * rng.random() ** (1.0 / 3.0)
        if r < r_min_inner:
            continue
        theta = np.arccos(2 * rng.random() - 1)
        phi = 2 * np.pi * rng.random()
        o = np.array([r * np.sin(theta) * np.cos(phi),
                      r * np.sin(theta) * np.sin(phi),
                      r * np.cos(theta)])
        if any(np.linalg.norm(o - np.array(p)) < r_min_pair for p in positions):
            continue

        rand_ax = rng.standard_normal(3)
        rand_ax /= np.linalg.norm(rand_ax)
        ang = rng.random() * 2 * np.pi
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        K = np.array([[0, -rand_ax[2], rand_ax[1]],
                      [rand_ax[2], 0, -rand_ax[0]],
                      [-rand_ax[1], rand_ax[0], 0]])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * K @ K

        half = 52.25 * np.pi / 180   # half of the 104.5 deg HOH angle
        h1_loc = np.array([0.96 * np.sin(half), 0, 0.96 * np.cos(half)])
        h2_loc = np.array([-0.96 * np.sin(half), 0, 0.96 * np.cos(half)])

        positions.append(o.tolist())
        symbols.append("O")
        positions.append((o + R @ h1_loc).tolist())
        symbols.append("H")
        positions.append((o + R @ h2_loc).tolist())
        symbols.append("H")
        placed += 1

    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    print(f"Built {cfg['label']} droplet: {len(atoms)} atoms, {placed} waters")
    if output_path:
        write(output_path, atoms)
        print(f"Saved {output_path}")
    return atoms


def main():
    p = argparse.ArgumentParser(description="La3+ metadynamics with MACE")
    p.add_argument("input", nargs="?", help="Input XYZ (for fresh run)")
    p.add_argument("--system", choices=["oh", "f"])
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--model-size", default="medium",
                   choices=["small", "medium", "large"])
    p.add_argument("--device", default=None)
    p.add_argument("--height", type=float, default=2.0, help="kJ/mol")
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--pace", type=int, default=500)
    p.add_argument("--bias-factor", type=float, default=15)
    p.add_argument("--temperature", type=float, default=300)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--ckpt-interval", type=int, default=5000)
    p.add_argument("--resume", metavar="SYSTEM")
    p.add_argument("--status", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--build", metavar="SYSTEM", choices=["oh", "f"])
    args = p.parse_args()

    if args.status:
        get_status(args.checkpoint_dir)
        return
    if args.analyze:
        analyze_and_plot(args.checkpoint_dir)
        return
    if args.build:
        out = Path(f"La_{'OH' if args.build == 'oh' else 'F'}_droplet.xyz")
        build_droplet(args.build, output_path=out)
        return

    if args.resume:
        result = load_checkpoint(args.resume, args.checkpoint_dir)
        if result is None:
            sys.exit(f"No checkpoint found for {args.resume}")
        atoms, metad, start_step = result
        atoms.pbc = False
        run_metadynamics(
            atoms, args.resume, args.steps,
            model_size=args.model_size, device=args.device,
            ckpt_dir=args.checkpoint_dir, ckpt_interval=args.ckpt_interval,
            start_step=start_step, metad=metad,
        )
        return

    if not args.input:
        p.error("Input XYZ required (or use --resume/--status/--analyze/--build)")
    if not args.system:
        p.error("--system (oh or f) required for fresh run")

    atoms = read(args.input)
    atoms.pbc = False
    print(f"Loaded {args.input}: {len(atoms)} atoms")
    run_metadynamics(
        atoms, args.system, args.steps,
        model_size=args.model_size, device=args.device,
        ckpt_dir=args.checkpoint_dir, ckpt_interval=args.ckpt_interval,
        sigma=args.sigma, height_kj=args.height, pace=args.pace,
        bias_factor=args.bias_factor, temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
