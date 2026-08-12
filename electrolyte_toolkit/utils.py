"""Shared helpers for the electrolyte MD toolkit. Everything else imports from here."""

import os

AVOGADRO = 6.02214076e23
AMU_TO_GRAMS = 1.66053906660e-24
ATM_TO_GPA = 1.01325e-4
ATM_TO_EV_A3 = 1.01325e-4 / 160.2176634  # via GPa, 160.2176634 GPa per eV/A^3

DEFAULT_MODEL = "orbmol_v2"
DEFAULT_TIMESTEP_FS = 1.0
DEFAULT_TRAJ_INTERVAL = 100
DEFAULT_PROP_INTERVAL = 10

ATOMIC_MASSES = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.380, "Br": 79.904,
    "I": 126.904, "Cs": 132.905, "Ba": 137.327,
}


def concentration_to_count(conc_mol_per_L: float, box_size_angstrom: float) -> int:
    """Turns a target molarity into a molecule count for a cubic box.

    N = c * L^3 * 6.022e-4, with L in angstroms and c in mol/L.
    """
    n = conc_mol_per_L * (box_size_angstrom ** 3) * 6.02214076e-4
    return max(1, round(n))


def total_mass_amu(elements: list[str]) -> float:
    """Sum atomic masses. Anything not in the table gets carbon's mass as a placeholder."""
    return sum(ATOMIC_MASSES.get(e, 12.0) for e in elements)


def total_mass_grams(elements: list[str]) -> float:
    """Same as total_mass_amu but in grams, which is what the density math wants."""
    return total_mass_amu(elements) * AMU_TO_GRAMS


def parse_pdb_elements(pdb_path: str) -> list[str]:
    """Pulls element symbols out of a PDB's ATOM/HETATM lines.

    Uses the element column when it's there. Plenty of files leave it blank, so
    fall back to guessing from the atom name.
    """
    elements = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            elem = ""
            if len(line) >= 78:
                elem = line[76:78].strip()
            if not elem:
                atom_name = line[12:16].strip()
                for i, ch in enumerate(atom_name):
                    if ch.isalpha():
                        candidate = atom_name[i:]
                        break
                else:
                    candidate = atom_name
                if len(candidate) >= 2 and candidate[:2] in ATOMIC_MASSES:
                    elem = candidate[:2]
                elif len(candidate) >= 1 and candidate[0] in ATOMIC_MASSES:
                    elem = candidate[0]
            if elem:
                elements.append(elem)
    return elements


def add_cryst1_to_pdb(pdb_path: str, box_size: float):
    """Sticks a CRYST1 record on a PDB so downstream tools know it's periodic.

    Overwrites the existing one if there already is one.
    """
    cryst1 = (
        f"CRYST1{box_size:9.3f}{box_size:9.3f}{box_size:9.3f}"
        f"  90.00  90.00  90.00 P 1           1\n"
    )
    with open(pdb_path) as f:
        content = f.read()
    if content.startswith("CRYST1"):
        lines = content.split("\n")
        lines[0] = cryst1.rstrip()
        content = "\n".join(lines)
    else:
        content = cryst1 + content
    with open(pdb_path, "w") as f:
        f.write(content)


def parse_molecule_spec(spec_str: str, box_size: float) -> tuple[str, str, int]:
    """Splits a 'name:path:amount' spec into its pieces.

    Amount is either a plain count or a number ending in M, which gets turned
    into a count for this box size.
    """
    parts = spec_str.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Molecule spec must be 'name:path:amount', got: {spec_str}"
        )
    name, path, amount = parts

    if amount.upper().endswith("M"):
        conc = float(amount[:-1])
        count = concentration_to_count(conc, box_size)
    else:
        count = int(amount)

    return name, path, count


def get_calculator(model: str = DEFAULT_MODEL, device: str | None = None):
    """Builds the ASE calculator for the given model.

    Supported models:
        'orbmol_v2'     OrbMol-v2 (Orbital Materials), trained on OMol25
        'uma'           UMA-s-1.2 (FAIRChem/Meta), trained on OMol25
        'mace_small'    MACE-MP-0 small (Materials Project)
        Any orb-models name (e.g. 'orb_v3_conservative_inf_omat')
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: No CUDA GPU detected, MD will be slow on CPU.")

    if model.startswith("uma"):
        try:
            from fairchem.core import pretrained_mlip, FAIRChemCalculator
        except ImportError:
            raise ImportError(
                "fairchem-core is not installed.\n"
                "  pip install fairchem-core\n"
                "  UMA checkpoints are gated, run: from huggingface_hub import login; login()"
            )
        uma_name = "uma-s-1p2" if model == "uma" else model.replace("_", "-")
        predictor = pretrained_mlip.get_predict_unit(uma_name, device=device)
        if hasattr(predictor, 'model'):
            predictor.model.use_checkpoint = False
        elif hasattr(predictor, 'inference_model'):
            predictor.inference_model.use_checkpoint = False
        calc = FAIRChemCalculator(predictor, task_name="omol")
        print(f"Calculator: FAIRChem UMA / {uma_name} on {device}")
        return calc

    if "orb" in model:
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.inference.calculator import ORBCalculator
        except ImportError:
            raise ImportError(
                "orb-models is not installed.\n"
                "  pip install orb-models\n"
                "  See https://github.com/orbital-materials/orb-models"
            )
        model_name = model.replace("-", "_")
        loader = getattr(pretrained, model_name, None)
        if loader is None:
            available = [a for a in dir(pretrained) if a.startswith("orb")]
            raise ValueError(
                f"Unknown model '{model_name}'. Available:\n  "
                + "\n  ".join(available)
            )
        orbff, atoms_adapter = loader(device=device, precision="float32-high")
        calc = ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)
        print(f"Calculator: orb-models / {model_name} on {device}")
        return calc

    if "mace" in model:
        try:
            from mace.calculators import mace_mp
        except ImportError:
            raise ImportError(
                "mace-torch is not installed.\n"
                "  pip install mace-torch"
            )
        calc = mace_mp(model=model, device=device, default_dtype="float64")
        print(f"Calculator: MACE / {model} on {device}")
        return calc

    raise ValueError(
        f"Unknown model: {model}\n"
        "Supported: 'orbmol_v2', 'uma', 'mace_small', or any orb-models name.\n"
        "Edit utils.get_calculator() to add more."
    )


class ProjectLayout:
    """One place for the directory structure, so the scripts aren't passing a dozen paths around.

        inputs/      Avogadro PDB files
        packed/      packed cell output
        nvt/         NVT equilibration (trajectory.traj, md.log, final.xyz)
        npt/         NPT equilibration
        anneal/      annealing equilibration
        analysis/    equilibration diagnostic plots
        vmd/         VMD-ready trajectory exports
    """

    SUBDIRS = ("inputs", "packed", "nvt", "npt", "anneal", "analysis", "vmd")

    def __init__(self, root: str):
        self.root = root

    @property
    def inputs(self) -> str:
        return os.path.join(self.root, "inputs")

    @property
    def packed_pdb(self) -> str:
        return os.path.join(self.root, "packed", "system.pdb")

    def equilibration_dir(self, protocol: str) -> str:
        return os.path.join(self.root, protocol)

    def trajectory(self, protocol: str) -> str:
        return os.path.join(self.root, protocol, "trajectory.traj")

    def md_log(self, protocol: str) -> str:
        return os.path.join(self.root, protocol, "md.log")

    def final_structure(self, protocol: str) -> str:
        return os.path.join(self.root, protocol, "final.xyz")

    @property
    def analysis(self) -> str:
        return os.path.join(self.root, "analysis")

    @property
    def vmd(self) -> str:
        return os.path.join(self.root, "vmd")

    def vmd_trajectory(self, fmt: str = "xyz") -> str:
        return os.path.join(self.root, "vmd", f"trajectory.{fmt}")

    def ensure_dirs(self):
        """Safe to call as many times as you want."""
        for d in self.SUBDIRS:
            os.makedirs(os.path.join(self.root, d), exist_ok=True)

    def summary(self) -> str:
        lines = [f"Project root: {self.root}"]
        for d in self.SUBDIRS:
            lines.append(f"  {d + '/':12s} -> {os.path.join(self.root, d)}")
        return "\n".join(lines)
