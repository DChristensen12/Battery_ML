# SolvationNet

This Repository is being used for machine learning simulations to study the molecular structure and dynamics of sodium ion battery electrolytes. It is part of my ongoing research at the Lawrence Berkeley National Lab under Nitesh Kumar and Samuel Blau.

This contains scripts and notebooks used during the research, the datasets produced by them, and other research outputs such as the poster below that was presented at the UC Berkeley College of Chemistry's 18th Annual Saegebarth Undergraduate Research Fair.

## Research Poster 

![Research Poster](Research/URF_Research_Poster.png)

## Dataset: Na-Ion Electrolyte Solvation Boxes

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/DChristensen12/na-ion-electrolyte-solvation-boxes)

Published on Hugging Face as a single repository with multiple subsets. Covers 9 sodium salts, 21 solvents/cosolvents, and concentrations from 0.1 M through 21 M across carbonate, ether, glyme, ionic liquid, phosphate, and aqueous systems. Both MLIPs were trained on the OMol25 dataset.

| Subset | Boxes | What it is |
|--------|-------|------------|
| `NPT_FAIRChem-UMA/` | 83 | NPT equilibrated (300 K, 1 atm) with UMA-s-1.2 (FAIRChem/Meta). Density-validated. |
| `NVT_OrbMolV2/` | 79 | NVT thermalized (50 ps, 300 K) with OrbMol-v2 (Orbital Materials). Fixed volume, no density adjustment. |
| `NVT_OrbMolV2/npt_extended/` | 4 | Subset of NVT boxes further equilibrated with NPT using OrbMol-v2. |
| `molecules/` | 32 | Individual molecule PDB files used as Packmol inputs. |

See the [dataset README](data/na-electrolyte-solvation-boxes/README.md) for the full file list, naming convention, and abbreviation tables.

## Deep-Shell

Will contain the scripts for lanthanide and actinide chemistry.

## Electrolyte Molecular Dynamics Toolkit

The `electrolyte_toolkit/` folder contains modular Python scripts for battery electrolyte molecular dynamics simulations. This is the work directly relating to the research poster and Na-Ion Electrolyte Solvation Boxes shown above.

### Pipeline

```
Input geometries (.pdb)
    │
    ▼
pack_cell.py          Pack molecules into a periodic box (Packmol)
    │
    ▼
equilibrate.py        NVT and/or NPT equilibration with checkpointing
    │
    ▼
analyze_trajectory.py   Diagnostic plots (T, PE, density vs time)
    │
    ▼
export_vmd.py         Export trajectory for VMD visualization
```

### Setup

#### Prerequisites

- Python 3.10+
- A CUDA GPU is strongly recommended (CPU works but is very slow for MD)
- [Packmol](https://m3g.github.io/packmol/) for cell packing

#### Quick Start

```bash
cd electrolyte_toolkit
./setup.sh
```

`setup.sh` creates a virtual environment, installs all Python dependencies, and
checks that Packmol and a GPU are available.

#### Manual Setup

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

Install Packmol separately (it's a compiled binary, not a Python package):

```bash
conda install -c conda-forge packmol
# or: sudo apt install packmol
```

#### Verify

```bash
source venv/bin/activate
python -c "import ase; import torch; print('OK')"
packmol < /dev/null   # should print Packmol banner, not "command not found"
```

### equilibrate.py

The main equilibration script. Supports multiple workflows and potentials with persistent checkpointing. If it gets interrupted, rerun the same command and it picks up where it left off. Only the latest checkpoint per box is kept.

```bash
# NPT-only with FAIRChem UMA (default)
python equilibrate.py my_box.xyz --model uma --workflow npt

# NVT+NPT with OrbMol-v2
python equilibrate.py my_box.xyz --model orbmol_v2 --workflow nvt+npt

# NVT only
python equilibrate.py my_box.xyz --model orbmol_v2 --workflow nvt

# Custom step counts
python equilibrate.py my_box.xyz --model uma --workflow npt --npt-steps 50000

# Resume an interrupted run
python equilibrate.py --resume my_box_name --model uma

# Check status of all boxes
python equilibrate.py --status
```

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--model` | `uma`, `orbmol_v2` | `uma` | ML potential |
| `--workflow` | `npt`, `nvt`, `nvt+npt` | `npt` | Equilibration protocol |
| `--nvt-steps` | integer | 50000 | NVT steps (50 ps at 1 fs) |
| `--npt-steps` | integer | 100000 | NPT steps (100 ps at 1 fs) |
| `--device` | `cuda`, `cpu` | `cuda` | Compute device |
| `--checkpoint-dir` | path | `~/electrolyte_equilibration` | Checkpoint storage |
| `--resume` | box name | none | Resume from last checkpoint |
| `--status` | flag | off | Print status of all boxes |

NPT parameters are set automatically per model. UMA gets `externalstress=1 bar`, `pfactor=0.1`, `ttime=100 fs` and an isotropic mask. OrbMol-v2 gets `externalstress` in eV/Å³, `pfactor=(75 fs)²·bulk_mod`, `ttime=25 fs`.

Live output includes a density convergence check (drift < 1% = converged).

### pack_cell.py

Packs molecules into a cubic periodic box using Packmol.

```bash
python pack_cell.py --project ./my-project \
  -m Na:inputs/Na.pdb:0.5M \
  -m PF6:inputs/PF6.pdb:0.5M \
  -m DME:inputs/DME.pdb:200 \
  --box-size 30
```

Amounts can be molar concentrations (`0.5M`) or explicit counts (`200`). Use `--dry-run` to preview without running, `--seed 42` for reproducible packing.

### run_md.py

Single-phase MD without checkpointing. Good for quick tests. Supports NVT, NPT, and annealing.

```bash
python run_md.py --project ./my-project -p npt -T 300 -P 1.0 -n 100000
python run_md.py --project ./my-project -p anneal --t-low 300 --t-high 500 --total-steps 200000 --num-cycles 5
```

### analyze_trajectory.py

Plots temperature, density, and energy vs time from a completed run.

```bash
python analyze_trajectory.py --project ./my-project --protocol npt
```

### export_vmd.py

Converts ASE trajectories to extended XYZ or multi-model PDB for VMD.

```bash
python export_vmd.py --project ./my-project --protocol npt --stride 10
```

### ML Potentials

The toolkit supports three ML potential families via `utils.get_calculator()`:

| Model | Install | Description |
|-------|---------|-------------|
| `uma` | `pip install fairchem-core` | UMA-s-1.2 (FAIRChem/Meta), trained on OMol25. HuggingFace login required. |
| `orbmol_v2` | `pip install orb-models` | OrbMol-v2 (Orbital Materials), trained on OMol25. Includes learnable electrostatics. |
| `mace_*` | `pip install mace-torch` | MACE-MP-0 (Materials Project). General-purpose, full periodic table. |

To use a specific model, pass `--model` to `equilibrate.py` or `run_md.py`. To add a new calculator, edit `get_calculator()` in `utils.py`.

### File Reference

| File | Purpose |
|------|---------|
| `utils.py` | Shared constants, calculator factory (UMA, OrbMol, MACE), project layout |
| `pack_cell.py` | Pack molecules into a periodic box (Packmol wrapper) |
| `equilibrate.py` | NVT/NPT equilibration with checkpointing and resume |
| `run_md.py` | Single-phase MD (NVT, NPT, annealing) without checkpointing |
| `analyze_trajectory.py` | Equilibration diagnostic plots |
| `export_vmd.py` | Trajectory export for VMD |
| `requirements.txt` | Python dependencies |
| `setup.sh` | One-command environment setup |

### Colab Notebooks

Google Colab notebooks for running equilibration on cloud GPUs with persistent checkpointing to Google Drive:

| Notebook | Workflow |
|----------|----------|
| `NVT_OrbMolV2_Equilibration.ipynb` | NVT + NPT with OrbMol-v2 |
| `NPT_FAIRChem-UMA_Equilibration.ipynb` | NPT-only with FAIRChem UMA |

Located in `notebooks/`. Upload input boxes, run cells, checkpoints save to Drive automatically. If runtime disconnects, reconnect and resume from the last checkpoint.

### Project Directory Layout

When using `--project`, scripts auto-derive paths from this structure:

```
my-project/
  inputs/       Avogadro PDB files (one per molecule)
  packed/       Packed cell output (system.pdb)
  nvt/          NVT equilibration (trajectory.traj, md.log, final.xyz)
  npt/          NPT equilibration
  anneal/       Annealing equilibration
  analysis/     Diagnostic plots (temperature, density, energy vs time)
  vmd/          VMD-ready trajectory exports (.xyz or .pdb)
```

You don't have to use `--project`. Every script also accepts explicit paths (`--input`, `--output`, etc.).
