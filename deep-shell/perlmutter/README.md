# La³⁺ Solvation MD on Perlmutter

10 ns unconstrained NVT MD on La³⁺ + anion + 128 H₂O droplets using
FAIRChem UMA OMOL (turbo inference) on NERSC Perlmutter A100 GPUs.

## Systems

| File | Composition | Atoms | Net Charge | Spin |
|------|-------------|-------|------------|------|
| La3+_F_droplet.xyz | 1 La³⁺ + 3 F⁻ + 128 H₂O | 388 | 0 | 1 (singlet) |
| La3+_OH_droplet.xyz | 1 La³⁺ + 3 OH⁻ + 128 H₂O | 391 | 0 | 1 |
| La3+_NO3_droplet.xyz | 1 La³⁺ + 3 NO₃⁻ + 128 H₂O | 397 | 0 | 1 |
| La3+_CO3_droplet.xyz | 1 La³⁺ + 2 CO₃²⁻ + 128 H₂O | 393 | **−1** | 1 |
| La3+_PO4_droplet.xyz | 1 La³⁺ + 1 PO₄³⁻ + 128 H₂O | 390 | 0 | 1 |

La³⁺ is [Xe]4f⁰ — closed-shell, no unpaired electrons; F⁻, OH⁻, NO₃⁻,
CO₃²⁻ and PO₄³⁻ are all closed-shell too, so every system is a singlet
(multiplicity 1). `charge` and `spin` in `atoms.info` are read by the
FAIRChem omol task head.

**CO₃ is not neutral.** 3+ cannot be balanced by an integer number of
divalent anions, so this system carries a net −1 and its electrostatics
are not strictly comparable to the other four. The clean fix is
2 La³⁺ + 3 CO₃²⁻, i.e. step 2 of the scale-up plan.

### Starting structures

The La³⁺ first shell is seeded as a **tricapped trigonal prism, CN = 9**,
with anions on the capping sites and water on the remaining vertices —
mean La–X of 2.48–2.55 Å against an experimental 2.54–2.56 Å
(Persson et al., *Chem. Eur. J.* **2008**, 14, 3056; D'Angelo et al.,
*Inorg. Chem.* **2011**, 50, 4572). First-shell water dipoles point away
from the cation (La···H ≥ 3.2 Å). Bulk water fills a ~9.5 Å sphere at
0.93–1.07× bulk density, and no intermolecular contact is shorter than
2.17 Å, so the droplets run at 1 fs without a minimization pass.

Rebuild them with:

```bash
python build_droplet.py -o droplets/
```

Anions start **inner-sphere and monodentate**. Nitrate and carbonate are
usually bidentate on La³⁺ in the solid state, so expect the first tens of
ps of each trajectory to be a re-coordination transient, not equilibrium.

## Quick start

```bash
# From SolvationNet repo root, deploy everything to Perlmutter:
bash deep-shell/perlmutter/deploy.sh <your_nersc_username>

# Then on Perlmutter:
cd $SCRATCH/solvationnet/deep-shell
bash perlmutter/setup_env.sh           # one-time
sbatch perlmutter/la_uma_md.sbatch     # submit
```

## Setup (one-time, on Perlmutter)

```bash
bash perlmutter/setup_env.sh

# UMA weights are gated — log in once:
python -c "from huggingface_hub import login; login()"
```

## Manual deploy (without deploy.sh)

```bash
# From local machine:
scp deep-shell/run_uma_md.py \
    user@perlmutter.nersc.gov:\$SCRATCH/solvationnet/deep-shell/

scp deep-shell/perlmutter/* \
    user@perlmutter.nersc.gov:\$SCRATCH/solvationnet/deep-shell/perlmutter/

scp droplets/La3+_*_droplet.xyz \
    user@perlmutter.nersc.gov:\$SCRATCH/solvationnet/deep-shell/droplets/
```

## Monitor

```bash
squeue -u $USER                          # job status
tail -f results/La3+_F/slurm.log         # live output for one system
tail -f la_uma_md_<jobid>.out            # SLURM wrapper output
```

## Retrieve results

```bash
# From local machine:
scp -r user@perlmutter.nersc.gov:\$SCRATCH/solvationnet/deep-shell/results/ .
```

## MD parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | UMA-s-1.2 (FAIRChem) | OMol25-trained, turbo inference (TF32 + torch.compile) |
| Ensemble | NVT (Langevin) | Weak friction 0.01/fs |
| Temperature | 300 K | Room temperature |
| Timestep | 1 fs | No bond constraints (MLIP handles H vibrations) |
| Duration | 10 ns | 10,000,000 steps |
| Trajectory | every 1 ps | ~10,000 frames |
| Checkpoints | every 50 ps | Resume-safe on timeout |
| Hardware | 1 node, 4× A100-80GB | all 5 systems concurrent; PO₄ shares GPU 0 with F |
| Wall time | 24 hours per job | **Not enough — expect multiple resubmissions, see below** |
| Account | M4292 | |

## Turbo inference

"Turbo" = three speedups stacked:
1. `torch.set_float32_matmul_precision("high")` — TF32 on A100 Tensor Cores
2. `use_checkpoint = False` — skip gradient checkpointing (training-only feature)
3. `torch.compile(model)` — fused CUDA kernels (if supported)

## Runtime expectations

10 ns at a 1 fs timestep is **10,000,000 UMA force calls per system**. At a
plausible 20–50 steps/s for ~390 atoms on an A100, that is roughly
**55–140 h of GPU time per system** — one 24 h allocation gets you somewhere
around 2–4 ns, and the five systems sharing four GPUs slows F and PO₄
further. Budget for **4–8 resubmissions**.

Benchmark before committing the queue time:

```bash
python run_uma_md.py droplets/La3+_F_droplet.xyz -o /tmp/bench --steps 2000
```

Read the `st/s` figure from the progress line and divide 10,000,000 by it.
If `torch.compile` is triggering repeated recompilation (the neighbour list
changes size every step), that number will be far worse than the un-compiled
baseline — compare with the `torch.compile` block in `run_uma_md.py`
disabled before deciding.

If 10 ns proves out of reach, the honest options are a shorter target
(2–5 ns is still well past the ~100 ps ligand-exchange timescale) or a
larger timestep with constrained H (rigid water), not a silently truncated
run.

## Resume on timeout

The sbatch script always passes `--resume`. If the job times out or OOMs,
resubmit the same script — each system picks up from its last checkpoint
(saved every 50 ps). No wasted compute, and because all five run
concurrently they advance together rather than one never starting.

```bash
sbatch perlmutter/la_uma_md.sbatch   # just resubmit
```

Note: checkpoints are written every 50 ps but the trajectory every 1 ps, so
each restart re-appends up to 50 frames that were already written. Dedupe on
the step column in `md.log` when analysing, or align the two intervals.

## Directory layout on Perlmutter

```
$SCRATCH/solvationnet/deep-shell/
├── run_uma_md.py              # MD runner
├── build_droplet.py           # Droplet builder
├── droplets/                  # Input structures
│   ├── La3+_F_droplet.xyz
│   └── ...
├── perlmutter/                # SLURM scripts
│   ├── la_uma_md.sbatch
│   ├── setup_env.sh
│   └── deploy.sh
└── results/                   # Output (created by sbatch)
    ├── La3+_F/
    │   ├── trajectory.xyz     # 10,000 frames
    │   ├── md.log             # step, time, T, PE, KE
    │   ├── checkpoint.pkl     # latest checkpoint
    │   ├── final.xyz          # final structure
    │   └── snap_*.xyz         # periodic snapshots
    └── ...
```
