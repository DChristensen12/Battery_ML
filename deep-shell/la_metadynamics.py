#!/usr/bin/env python3
"""Well-tempered metadynamics of La3+ coordination with MACE-polar.

Runs inside an Axon Modal sandbox (chemistry-py-3.12-gpu, L40S).
All I/O goes to /mnt/remote so it persists across sandbox restarts.

Usage (from Axon sandbox):
    TAG="oh" python /mnt/remote/<project>/run_metad.py oh
    TAG="f"  python /mnt/remote/<project>/run_metad.py f

Dependencies (install in sandbox before running):
    pip install mace-torch
    pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0

Versions used:
    mace-torch 0.3.16, graph_electrostatics v0.4.0, ASE 3.26.0, PyTorch 2.9.1
"""

import numpy as np
import os, sys, time, pickle
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.constraints import FixCom
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import mace_polar

TAG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TAG", "oh")
CONFIGS = {
    "oh": {"input": "La_OH_droplet.xyz", "anions": [1, 3, 5], "r0": 3.5},
    "f":  {"input": "La_F_droplet.xyz",  "anions": [1, 2, 3], "r0": 3.2},
}
cfg = CONFIGS[TAG]

OUT = os.environ.get("METAD_DIR", "/mnt/remote/26-08-20-la-metadynamics")
INPUT = os.path.join(OUT, cfg["input"])

LA = 0; TOTAL = 400_000; DT = 1.0; TEMP = 300; FRIC = 0.01
SIG = 0.15; H0_EV = 2.0 / 96.485; GAMMA = 15; PACE = 500
NN, MM = 6, 12; LOG_INT = 500; CKPT_INT = 2000; SNAP_INT = 25000

HILLS_PATH = os.path.join(OUT, f"hills_{TAG}.dat")
COLVAR_PATH = os.path.join(OUT, f"colvar_{TAG}.dat")
LOG_PATH = os.path.join(OUT, f"metad_{TAG}.log")
CKPT_PATH = os.path.join(OUT, f"ckpt_{TAG}.pkl")


def cn_and_grad(pos, la, anions, r0, nn, mm):
    n = len(pos); cn = 0.0; grad = np.zeros((n, 3))
    la_pos = pos[la]
    for ai in anions:
        d = pos[ai] - la_pos; r = np.linalg.norm(d)
        if r < 1e-10: continue
        x = r / r0; xn = x**nn; xm = x**mm
        num = 1.0 - xn; den = 1.0 - xm
        if abs(den) < 1e-30: continue
        cn += num / den
        dnum = -nn * x**(nn-1) / r0; dden = -mm * x**(mm-1) / r0
        dsdr = (dnum * den - num * dden) / (den * den)
        drdx = d / r; grad[ai] += dsdr * drdx; grad[la] -= dsdr * drdx
    return cn, grad


class MetadCalc(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, base, la, anions, r0, nn, mm, sig, h0, gamma, pace, kbt):
        super().__init__()
        self.base = base; self.la = la; self.anions = anions; self.r0 = r0
        self.nn = nn; self.mm = mm; self.sig = sig
        self.h0 = h0; self.gamma = gamma; self.pace = pace; self.kbt = kbt
        self.hills = []
        self._cn = 0.0; self._bias = 0.0; self._base_e = 0.0; self._temp = 0.0

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.base.calculate(atoms, properties, all_changes)
        e = self.base.results['energy']; f = self.base.results['forces'].copy()
        pos = atoms.positions
        cn, cn_grad = cn_and_grad(pos, self.la, self.anions, self.r0, self.nn, self.mm)
        bias = 0.0; dbias = 0.0
        for cn_k, w_k in self.hills:
            g = np.exp(-(cn - cn_k)**2 / (2 * self.sig**2))
            bias += w_k * g; dbias += w_k * g * (-(cn - cn_k) / self.sig**2)
        self.results['energy'] = e + bias
        self.results['forces'] = f - dbias * cn_grad
        self._cn = cn; self._bias = bias; self._base_e = e
        ke = atoms.get_kinetic_energy()
        self._temp = 2 * ke / ((3 * len(atoms) - 3) * units.kB)

    def deposit(self, cn):
        v = sum(w * np.exp(-(cn - c)**2 / (2 * self.sig**2)) for c, w in self.hills) if self.hills else 0.0
        w = self.h0 * np.exp(-v / (self.kbt * (self.gamma - 1)))
        self.hills.append((cn, w)); return w


atoms = read(INPUT)
atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.info["external_field"] = [0.0, 0.0, 0.0]

print(f"[{TAG.upper()}] Loading MACE-polar (polar-1-s) on CUDA...", flush=True)
base = mace_polar(model="polar-1-s", device="cuda", default_dtype="float32")
kbt = units.kB * TEMP
metad = MetadCalc(base, LA, cfg["anions"], cfg["r0"], NN, MM, SIG, H0_EV, GAMMA, PACE, kbt)
atoms.calc = metad; atoms.set_constraint(FixCom())

start_step = 0
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH, "rb") as fh: ckpt = pickle.load(fh)
    atoms.positions = ckpt["positions"]; atoms.set_velocities(ckpt["velocities"])
    metad.hills = ckpt["hills"]; start_step = ckpt["step"]
    print(f"[{TAG.upper()}] Resumed from step {start_step}, {len(metad.hills)} hills", flush=True)
else:
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMP)
    print(f"[{TAG.upper()}] Fresh start, {len(atoms)} atoms", flush=True)

dyn = Langevin(atoms, timestep=DT * units.fs, temperature_K=TEMP, friction=FRIC / units.fs)
mode = "a" if start_step > 0 else "w"
hills_fh = open(HILLS_PATH, mode); colvar_fh = open(COLVAR_PATH, mode); log_fh = open(LOG_PATH, mode)
if start_step == 0:
    hills_fh.write("# step cn weight_eV\n")
    colvar_fh.write("# step cn bias_eV energy_eV temperature_K\n")

print(f"[{TAG.upper()}] Starting metadynamics: {TOTAL} steps, R0={cfg['r0']}, sigma={SIG}, gamma={GAMMA}", flush=True)
t0 = time.time(); step = start_step

while step < TOTAL:
    dyn.run(1); step += 1
    if step % PACE == 0:
        w = metad.deposit(metad._cn)
        hills_fh.write(f"{step} {metad._cn:.6f} {w:.10e}\n"); hills_fh.flush()
    if step % LOG_INT == 0:
        elapsed = time.time() - t0
        speed = (step - start_step) / elapsed if elapsed > 0 else 0
        colvar_fh.write(f"{step} {metad._cn:.6f} {metad._bias:.10e} {metad._base_e:.6f} {metad._temp:.2f}\n")
        colvar_fh.flush()
        log_fh.write(f"step={step} cn={metad._cn:.4f} bias={metad._bias:.6f} E={metad._base_e:.4f} T={metad._temp:.1f} speed={speed:.2f}\n")
        log_fh.flush()
    if step % 10000 == 0:
        elapsed = time.time() - t0; speed = (step - start_step) / elapsed if elapsed > 0 else 0
        print(f"[{TAG.upper()}] step={step}/{TOTAL} ({step/10:.0f}ps) cn={metad._cn:.3f} T={metad._temp:.0f}K {speed:.1f}st/s hills={len(metad.hills)}", flush=True)
    if step % SNAP_INT == 0:
        write(os.path.join(OUT, f"metad_{TAG}_step{step}.xyz"), atoms)
    if step % CKPT_INT == 0:
        ckpt_data = {"positions": atoms.positions.copy(), "velocities": atoms.get_velocities().copy(),
                     "hills": list(metad.hills), "step": step}
        tmp = CKPT_PATH + ".tmp"
        with open(tmp, "wb") as fh: pickle.dump(ckpt_data, fh)
        os.replace(tmp, CKPT_PATH)

write(os.path.join(OUT, f"metad_{TAG}_final.xyz"), atoms)
hills_fh.close(); colvar_fh.close(); log_fh.close()
print(f"[{TAG.upper()}] COMPLETED: {step} steps, {len(metad.hills)} hills, {(time.time()-t0)/3600:.1f}h", flush=True)
