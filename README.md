# SolvationNet

This Repository is being used for machine learning simulations to study the molecular structure and dynamics of sodium ion battery electrolytes. It is part of my ongoing research at the Lawrence Berkeley National Lab under Nitesh Kumar and Samuel Blau.

## Research Poster 

![Research Poster](Research/URF_Research_Poster.png)

## Datasets

### Dataset 1:

Initial configurations are packed with Packmol and geometry-optimized with the FIRE algorithm. Systems are then equilibrated in two stages: NVT Langevin dynamics (50 ps, 300 K) to thermalize the structure, followed by NPT dynamics (100 ps, 300 K, 1 atm) to relax the box volume to equilibrium density. 
Both stages used the OrbMol-v2 potential (Orbital Materials), trained on the OMol25 dataset.


### Dataset 2:

Configurations equilibrated under NPT with OPLS-AA and the OMol25-trained UMA potential (which is FAIRChem)

### Dataset 3:

Initial configurations were packed with Packmol and geometry-optimized with the FIRE algorithm. Systems were then equilibrated with NPT dynamics (100 ps, 300 K, 1 atm) to relax the box volume to equilibrium density. The run used the UMA potential (FAIRChem), trained on the OMol25 dataset.

## Electrolyte MD Toolkit

The electrolyte_toolkit folder in SolvationNet contains modular Python scripts for battery electrolyte molecular dynamics simulations. In particular, this is the work directly relating to the research poster shown above (adaptation of the `SIB.ipynb` and uses data from `data/geometries`).

It takes geometry-optimized PDB files from Avogadro (you can make the files elsewhere, it only has to be in a .pdb format when used here), packs them into a simulation
cell, runs equilibration with an ML potential, checks convergence, and exports
trajectories for VMD visualization (or wherever you'd want to visualize the trajectory file also works).

```
Avogadro (.pdb) --> pack_cell.py --> run_md.py --> analyze_trajectory.py --> export_vmd.py --> VMD
```

### Setup

#### Prerequisites

- Python 3.10+
- A CUDA GPU is strongly recommended (CPU works but is very slow for MD)
- [Packmol](https://m3g.github.io/packmol/) for cell packing

#### Quick Start

```bash
# Clone or copy this directory, then:
cd electrolyte_toolkit
./setup.sh
```

`setup.sh` creates a virtual environment, installs all Python dependencies, and
checks that Packmol and a GPU are available.

#### Manual Setup

If you prefer to set things up yourself:

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

### Project Directory Layout

Every run uses a **project directory** that keeps inputs and outputs organized.
Pass `--project <dir>` to any script and it auto-derives all file paths:

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

You don't have to use `--project`. Every script also accepts explicit paths
(`--input`, `--output`, etc.) if you want to organize differently.

### Pipeline Walkthrough

#### Example: 1 M NaPF6 in 1,2-DME

##### 1. Prepare Input Structures

Optimize each molecule's geometry in Avogadro and export as PDB. Place
the files in your project's `inputs/` folder:

```
my-project/inputs/Na.pdb
my-project/inputs/PF6.pdb
my-project/inputs/DME.pdb
```

##### 2. Pack the Simulation Cell

```bash
python pack_cell.py --project ./my-project \
  -m Na:my-project/inputs/Na.pdb:0.5M \
  -m PF6:my-project/inputs/PF6.pdb:0.5M \
  -m DME:my-project/inputs/DME.pdb:200 \
  --box-size 30
```

- `-m name:path:amount` specifies each molecule. Amount is either:
  - A molar concentration like `0.5M` (script computes molecule count from box volume)
  - An explicit integer count like `200`
- `--box-size 30` creates a 30 x 30 x 30 angstrom cubic cell
- Output: `my-project/packed/system.pdb` with a CRYST1 record for periodic boundaries

Use `--dry-run` to preview the Packmol input without running it.
Use `--seed 42` for reproducible packing.

##### 3. Run Equilibration

Choose one of three protocols:

**NVT** (constant volume + temperature):
```bash
python run_md.py --project ./my-project -p nvt \
  -T 300 -n 50000
```

**NPT** (constant pressure + temperature):
```bash
python run_md.py --project ./my-project -p npt \
  -T 300 -P 1.0 -n 100000
```

**Annealing** (cyclic heating/cooling):
```bash
python run_md.py --project ./my-project -p anneal \
  --t-low 300 --t-high 500 \
  --total-steps 200000 --num-cycles 5
```

Each protocol produces three files in its output directory (e.g. `npt/`):

| File | Contents |
|------|----------|
| `trajectory.traj` | Full atomic trajectory (positions, cell, velocities) |
| `md.log` | Property time series (time, energy, temperature) |
| `final.xyz` | Last frame of the simulation |

Key options:
- `-T` / `--temperature`: temperature in Kelvin (default: 300)
- `-P` / `--pressure`: pressure in atm for NPT (default: 1.0)
- `-n` / `--steps`: number of MD steps (NVT/NPT)
- `-dt` / `--timestep`: timestep in femtoseconds (default: 1.0)
- `--model`: ML potential model name (default: `orb_v3_conservative_omol`)
- `--device`: `cuda` or `cpu` (auto-detected if omitted)
- `--force`: overwrite previous outputs (safe re-runs)

##### 4. Check Equilibration

```bash
python analyze_trajectory.py --project ./my-project --protocol npt
```

Generates three PNG plots in `my-project/analysis/`:

- **temperature_vs_time.png**: should flatten to a stable mean
- **density_vs_time.png**: should converge/be constant (meaningful for NPT; constant for NVT)
- **energy_vs_time.png**: potential energy should plateau

Each plot includes a running average overlay and the script prints summary
statistics (mean, std, drift) for the last 25% of the trajectory.

If the system hasn't equilibrated, run more steps or try annealing first.

##### 5. Export for VMD

```bash
python export_vmd.py --project ./my-project --protocol npt --stride 10
```

Converts the ASE trajectory to extended XYZ (default) or multi-model PDB:

```bash
# XYZ (default, recommended):
python export_vmd.py --project ./my-project --protocol npt

# PDB format:
python export_vmd.py --project ./my-project --protocol npt -f pdb
```

`--stride 10` writes every 10th frame to reduce file size.

Open in VMD:
```bash
vmd my-project/vmd/trajectory.xyz
```

### ML Potential

The default potential is **orb-v3-conservative-omol** from
[Orbital Materials](https://github.com/orbital-materials/orb-models), which was
trained on the OMol dataset of organic molecules and is well-suited for battery
electrolyte environments. I set this so that it wouldn't randomly break if not presented with a ML Potential.

To use a different potential, pass `--model` to `run_md.py`:

```bash
# MACE foundation model (install: pip install mace-torch)
python run_md.py --project ./my-project -p nvt -T 300 -n 50000 \
  --model mace_mp

# Different Orb checkpoint
python run_md.py --project ./my-project -p nvt -T 300 -n 50000 \
  --model orb_v3_conservative_inf_omat
```

To plug in a completely different ASE calculator, edit `get_calculator()` in
`utils.py`.

### File Reference

| File | Purpose |
|------|---------|
| `utils.py` | Shared constants, atomic masses, PDB parsing, calculator factory, project layout |
| `pack_cell.py` | Pack molecules into a cubic cell (wraps Packmol) |
| `run_md.py` | Run MD equilibration: NVT, NPT, or annealing (ASE + ML potential) |
| `analyze_trajectory.py` | Plot temperature, density, and energy vs time |
| `export_vmd.py` | Convert ASE trajectory to XYZ/PDB for VMD |
| `requirements.txt` | Python dependencies |
| `setup.sh` | One-command environment setup |

### Typical Equilibration Workflow

For a new electrolyte system, a common strategy is:

1. Anneal first to escape bad initial packing. Heat to 500 K and cool back
   to 300 K over several cycles.
2. Run NPT equilibration at the target temperature and 1 atm to let the density
   converge
3. Check the density and temperature plots. If they've plateaued, the system
   is equilibrated.
4. Use the `final.xyz` from the NPT run as input for production MD or further
   analysis

# Current Status/Updates:

Currently equilibrating boxes of mixtures that I found to be of interest via literature searchs. I also am going to make some
changes with the scripts and ensure it runs more smoothly. The resulting dataset will also be uploaded to hugging face.
There will be two datasets created!

Stay tuned!