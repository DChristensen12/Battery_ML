#!/usr/bin/env python3
"""Build La3+ + anion + 128 water droplets for UMA OMOL MD.

One non-periodic extended-xyz file per anion system:

  La3+_F_droplet.xyz     1 La3+ + 3 F-     + 128 H2O  (388 atoms, net  0)
  La3+_OH_droplet.xyz    1 La3+ + 3 OH-    + 128 H2O  (391 atoms, net  0)
  La3+_NO3_droplet.xyz   1 La3+ + 3 NO3-   + 128 H2O  (397 atoms, net  0)
  La3+_CO3_droplet.xyz   1 La3+ + 2 CO3^2- + 128 H2O  (393 atoms, net -1)
  La3+_PO4_droplet.xyz   1 La3+ + 1 PO4^3- + 128 H2O  (390 atoms, net  0)

Carbonate is the odd one out. You can't balance 3+ with a whole number of
divalent anions, so that droplet carries a net -1. If you need it neutral,
go to 2 La3+ + 3 CO3^2-.

The first shell gets built as a tricapped trigonal prism with the anions on
the capping sites, then the rest of the droplet fills in off a grid. Water
orientations are chosen to keep hydrogens off each other, so these are safe
to hand straight to 1 fs MD without minimizing first.

Usage:
  python build_droplet.py
  python build_droplet.py -o /path/to/output
  python build_droplet.py --seed 42
"""

import argparse
import math
import os

import numpy as np
from ase import Atoms
from ase.io import write


R_OH = 0.9572       # water O-H bond, A
THETA_HOH = 104.52  # water H-O-H angle, degrees
LA_CHARGE = 3

N_WATER = 128
GRID_SPACING = 2.9   # A, candidate O-site grid
SPHERE_RADIUS = 12.0 # A, outer edge of the droplet
MIN_DIST_CORE = 2.5  # A, water O to any core atom
MIN_DIST_OO = 2.3    # A, water O to water O

# oxygens are pinned to grid sites, so hydrogens are the only thing left that
# can clash. MIN_H_HEAVY sits under the ~1.9 A of a real O-H...O hydrogen bond,
# otherwise we'd throw away H-bonded geometries along with the bad ones.
MIN_H_H = 1.75         # A, H to H on another molecule
MIN_H_HEAVY = 1.70     # A, H to any heavy atom on another molecule
MIN_H_LA = 2.90        # A, keeps water dipoles pointed away from the cation
NEIGHBOR_CUTOFF = 6.0  # A, how far a water looks for neighbours
ROT_TRIES = 256        # orientations tried per bulk water
SPIN_TRIES = 72        # spins about the La-O axis for a first-shell water
REFINE_SWEEPS = 4

# aqueous La3+ is nine-coordinate in a tricapped trigonal prism: six prism
# oxygens near 2.52 A and three caps near 2.64 A. A bare cubic grid hands you
# CN 6 at 3.0 A instead, which is a lattice artifact, not chemistry. Seed the
# real motif and let MD take it from there.
#   Persson et al., Chem. Eur. J. 2008, 14, 3056  (EXAFS/LAXS, CN = 9)
#   D'Angelo et al., Inorg. Chem. 2011, 50, 4572  (La-O 2.54-2.56 A)
R_PRISM = 2.52
R_CAP = 2.64
MIN_DIST_SHELL = 2.80  # A, grid water O to any first-shell O

# each template puts the coordinating atom at the origin with the rest of the
# molecule running along +z. Placing one rotates +z onto the La-to-anion
# direction, then slides it out to la_dist.
ANIONS = {
    "F": {
        "symbols": ["F"],
        "coords": [[0.0, 0.0, 0.0]],
        "charge": -1,
        "la_dist": 2.40,
    },
    "OH": {
        "symbols": ["O", "H"],
        "coords": [
            [0.0, 0.0, 0.0],    # O (faces La)
            [0.0, 0.0, 0.97],   # H (away from La)
        ],
        "charge": -1,
        "la_dist": 2.55,
    },
    "NO3": {
        # trigonal planar, N-O = 1.26 A
        "symbols": ["O", "N", "O", "O"],
        "coords": [
            [0.000,  0.000, 0.000],  # coord O
            [0.000,  0.000, 1.260],  # N
            [1.091,  0.000, 1.890],  # O
            [-1.091, 0.000, 1.890],  # O
        ],
        "charge": -1,
        "la_dist": 2.55,
    },
    "CO3": {
        # trigonal planar, C-O = 1.28 A
        "symbols": ["O", "C", "O", "O"],
        "coords": [
            [0.000,  0.000, 0.000],
            [0.000,  0.000, 1.280],
            [1.109,  0.000, 1.920],
            [-1.109, 0.000, 1.920],
        ],
        "charge": -2,
        "la_dist": 2.55,
    },
    "PO4": {
        # tetrahedral, P-O = 1.54 A
        "symbols": ["O", "P", "O", "O", "O"],
        "coords": [
            [0.000,  0.000,  0.000],  # coord O
            [0.000,  0.000,  1.540],  # P
            [1.452,  0.000,  2.053],  # O
            [-0.726, 1.258,  2.053],  # O
            [-0.726, -1.258, 2.053],  # O
        ],
        "charge": -3,
        "la_dist": 2.55,
    },
}


def rotation_z_to(d):
    """Rodrigues rotation mapping [0,0,1] onto unit vector d."""
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(d, z):
        return np.eye(3)
    if np.allclose(d, -z):
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, d)
    s = np.linalg.norm(v)
    c = np.dot(z, d)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def distribute_directions(n):
    """n approximately-evenly-spaced unit vectors on a sphere."""
    if n == 1:
        return [np.array([1.0, 0.0, 0.0])]
    if n == 2:
        return [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
    if n == 3:
        return [
            np.array([math.cos(a), math.sin(a), 0.0])
            for a in [0, 2 * math.pi / 3, 4 * math.pi / 3]
        ]
    # Fibonacci sphere for n > 3
    golden = (1 + math.sqrt(5)) / 2
    dirs = []
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden
        dirs.append(np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]))
    return dirs


def random_rotations(rng, k):
    """k uniform random SO(3) rotations, shape (k, 3, 3) (Shoemake 1992)."""
    u1, u2, u3 = rng.random((3, k))
    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    R = np.empty((k, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def ttp_directions():
    """Unit vectors of a tricapped trigonal prism, as (6 prism, 3 caps).

    The two prism triangles sit at z = +/-h with azimuths 0/120/240, and the
    caps lie in the z = 0 plane staggered 60 degrees off those. Setting the
    triangle edge equal to the prism height is what pins it to the D3h shape,
    and that works out to rho = 2/sqrt(7), h = sqrt(3/7) on the unit sphere.
    """
    rho = 2.0 / math.sqrt(7.0)
    hgt = math.sqrt(3.0 / 7.0)
    prism = [
        np.array([math.cos(phi) * rho, math.sin(phi) * rho, zs * hgt])
        for phi in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)
        for zs in (1.0, -1.0)
    ]
    caps = [
        np.array([math.cos(phi), math.sin(phi), 0.0])
        for phi in (math.pi / 3, math.pi, 5 * math.pi / 3)
    ]
    return prism, caps


def anion_directions(n_anions):
    """Where the inner-sphere anions go.

    Three or fewer take the TTP caps. More than that only comes up once you
    add a second La3+, and there we just spread them over a sphere.
    """
    _, caps = ttp_directions()
    if n_anions <= len(caps):
        return caps[:n_anions]
    return distribute_directions(n_anions)


def first_shell_waters(n_anions):
    """O positions for whichever TTP sites the anions didn't take."""
    prism, caps = ttp_directions()
    sites = [d * R_PRISM for d in prism]
    sites += [d * R_CAP for d in caps[n_anions:]]
    return np.array(sites)


def place_anions(anion_key, n_anions):
    """n_anions copies of anion_key arranged around La, which sits at the origin."""
    anion = ANIONS[anion_key]
    template_syms = anion["symbols"]
    template_pos = np.array(anion["coords"])
    la_dist = anion["la_dist"]

    directions = anion_directions(n_anions)
    symbols, positions = [], []
    for d in directions:
        R = rotation_z_to(d)
        for sym, pos in zip(template_syms, template_pos):
            symbols.append(sym)
            positions.append(R @ pos + la_dist * np.asarray(d))
    return symbols, positions


def select_oxygen_sites(core_positions, shell_positions, n_water):
    """Pick n_water bulk water oxygens off a grid, closest to La first.

    Anything crowding a core atom or a first-shell water gets dropped.
    """
    core = np.asarray(core_positions, dtype=float).reshape(-1, 3)
    shell = np.asarray(shell_positions, dtype=float).reshape(-1, 3)

    n = int(2 * SPHERE_RADIUS / GRID_SPACING) + 1
    lin = np.linspace(-SPHERE_RADIUS, SPHERE_RADIUS, n)
    candidates = []
    for xi in lin:
        for yi in lin:
            for zi in lin:
                p = np.array([xi, yi, zi])
                if np.linalg.norm(p) > SPHERE_RADIUS:
                    continue
                if len(core) > 0 and np.min(np.linalg.norm(core - p, axis=1)) < MIN_DIST_CORE:
                    continue
                if len(shell) > 0 and np.min(np.linalg.norm(shell - p, axis=1)) < MIN_DIST_SHELL:
                    continue
                candidates.append(p)

    candidates.sort(key=lambda p: np.linalg.norm(p))

    selected = []
    for p in candidates:
        if len(selected) >= n_water:
            break
        if all(np.linalg.norm(p - s) >= MIN_DIST_OO for s in selected):
            selected.append(p)

    if len(selected) < n_water:
        raise RuntimeError(
            f"Only placed {len(selected)}/{n_water} waters. "
            f"Increase SPHERE_RADIUS or decrease GRID_SPACING."
        )
    return np.array(selected)


def _limits(symbols):
    """Closest an H is allowed to get to each environment atom."""
    return np.array([
        MIN_H_H if s == "H" else MIN_H_LA if s == "La" else MIN_H_HEAVY
        for s in symbols
    ])


def _clearance(h_pos, env_pos, env_limit):
    """How much room candidate H positions leave against their surroundings.

    h_pos is (k, 2, 3), the two hydrogens for each of k candidate orientations.
    env_pos is (m, 3) with the per-atom minimum in env_limit. Comes back as
    (k,), the worst margin for each candidate, so >= 0 means nothing is close.
    """
    if len(env_pos) == 0:
        return np.full(len(h_pos), np.inf)
    d = np.linalg.norm(h_pos[:, :, None, :] - env_pos[None, None, :, :], axis=-1)
    return (d - env_limit[None, None, :]).min(axis=(1, 2))


def orient_waters(o_sites, core_syms, core_pos, rng, n_shell=0):
    """Turn each water so its hydrogens stay clear of everything around it.

    The oxygens are already fixed, so this is only ever a search over
    rotations. The first n_shell waters are bound to La3+, so their dipoles
    are held pointing straight out from it and only the spin about the La-O
    axis gets searched. Everything else gets a free SO(3) search. One greedy
    pass working outward from the cation, then REFINE_SWEEPS relaxation
    sweeps once every water is down.
    """
    core_pos = np.asarray(core_pos, dtype=float).reshape(-1, 3)
    core_limit = _limits(core_syms)

    half = math.radians(THETA_HOH / 2)
    h_ref = np.array([
        [0.0, R_OH * math.sin(half), R_OH * math.cos(half)],
        [0.0, -R_OH * math.sin(half), R_OH * math.cos(half)],
    ])

    n = len(o_sites)
    h_pos = o_sites[:, None, :] + h_ref[None, :, :]

    def candidate_rotations(i):
        if i < n_shell:
            u = o_sites[i] / np.linalg.norm(o_sites[i])
            base = rotation_z_to(u)          # +z (the dipole) -> radially outward
            angles = np.linspace(0, 2 * math.pi, SPIN_TRIES, endpoint=False)
            K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            spins = (np.eye(3)[None] + np.sin(angles)[:, None, None] * K[None]
                     + (1 - np.cos(angles))[:, None, None] * (K @ K)[None])
            return spins @ base
        return random_rotations(rng, ROT_TRIES)

    # neighbour lists are static, the oxygens never move
    oo = np.linalg.norm(o_sites[:, None, :] - o_sites[None, :, :], axis=-1)
    np.fill_diagonal(oo, np.inf)
    water_nbrs = [np.flatnonzero(oo[i] < NEIGHBOR_CUTOFF) for i in range(n)]
    oc = np.linalg.norm(o_sites[:, None, :] - core_pos[None, :, :], axis=-1)
    core_nbrs = [np.flatnonzero(oc[i] < NEIGHBOR_CUTOFF) for i in range(n)]

    def environment(i, placed):
        """core neighbours plus the O and H of nearby waters"""
        w = water_nbrs[i]
        if placed is not None:
            w = w[np.isin(w, placed)]
        c = core_nbrs[i]
        pos = np.concatenate([core_pos[c], o_sites[w], h_pos[w].reshape(-1, 3)])
        limit = np.concatenate([
            core_limit[c],
            np.full(len(w), MIN_H_HEAVY),
            np.full(2 * len(w), MIN_H_H),
        ])
        return pos, limit

    def best_orientation(i, placed):
        env_pos, env_limit = environment(i, placed)
        R = candidate_rotations(i)
        cand = o_sites[i] + np.einsum("kab,jb->kja", R, h_ref)  # (k, 2, 3)
        score = _clearance(cand, env_pos, env_limit)
        k = int(np.argmax(score))
        return cand[k], float(score[k])

    # greedy pass, innermost water first since o_sites is sorted by distance
    for i in range(n):
        h_pos[i], _ = best_orientation(i, placed=np.arange(i))

    # now that every water is down, retry each one against the full environment
    # and keep whichever orientation is actually better
    for _ in range(REFINE_SWEEPS):
        worst = np.inf
        for i in range(n):
            env_pos, env_limit = environment(i, placed=None)
            current = _clearance(h_pos[i][None], env_pos, env_limit)[0]
            cand, score = best_orientation(i, placed=None)
            if score > current:
                h_pos[i] = cand
                current = score
            worst = min(worst, current)
        if worst >= 0.0:
            break

    symbols, positions = [], []
    for i in range(n):
        symbols.extend(["O", "H", "H"])
        positions.extend([o_sites[i], h_pos[i][0], h_pos[i][1]])
    return symbols, positions


def min_intermolecular_contact(symbols, positions, n_core):
    """Shortest distance between atoms of different molecules, in A."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(pos)
    # the core counts as a single molecule, it's all bound to La. then one id
    # per water
    mol = np.zeros(n, dtype=int)
    mol[n_core:] = 1 + (np.arange(n - n_core) // 3)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    d[mol[:, None] == mol[None, :]] = np.inf
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return float(d[i, j]), f"{symbols[i]}-{symbols[j]}"


def coordination_number(symbols, positions, n_core, cutoff=3.2):
    """First-shell CN of La3+: donors (O, F) inside cutoff, plus the mean La-X."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    d = np.linalg.norm(pos - pos[0], axis=1)
    donors = [i for i in range(1, len(pos))
              if symbols[i] in ("O", "F") and d[i] < cutoff]
    return len(donors), float(np.mean(d[donors])) if donors else 0.0


def build_system(anion_key, seed=42):
    """Assemble 1 La3+ + charge-balanced anions + 128 H2O."""
    rng = np.random.default_rng(seed)
    anion = ANIONS[anion_key]

    n_anions = math.ceil(LA_CHARGE / abs(anion["charge"]))
    net_charge = LA_CHARGE + n_anions * anion["charge"]

    symbols = ["La"]
    positions = [np.array([0.0, 0.0, 0.0])]

    anion_syms, anion_pos = place_anions(anion_key, n_anions)
    symbols.extend(anion_syms)
    positions.extend(anion_pos)

    n_core = len(symbols)

    # o_sites has to stay sorted by distance from La, orient_waters walks it
    # outward and that ordering is what makes the greedy pass work
    shell_sites = first_shell_waters(n_anions)
    grid_sites = select_oxygen_sites(positions, shell_sites,
                                     N_WATER - len(shell_sites))
    o_sites = np.vstack([shell_sites, grid_sites])

    water_syms, water_pos = orient_waters(o_sites, symbols, positions, rng,
                                          n_shell=len(shell_sites))
    symbols.extend(water_syms)
    positions.extend(water_pos)

    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    atoms.info["charge"] = int(net_charge)
    atoms.info["spin"] = 1

    contact, pair = min_intermolecular_contact(symbols, positions, n_core)
    cn = coordination_number(symbols, positions, n_core)
    return atoms, n_anions, int(net_charge), contact, pair, cn


def main():
    parser = argparse.ArgumentParser(
        description="Build La³⁺ + anion water droplets for UMA OMOL MD"
    )
    parser.add_argument("-o", "--output-dir", default=".",
                        help="directory for output .xyz files (default: cwd)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for water orientations (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building La³⁺ + anion + 128 H₂O droplets\n")
    for key in ANIONS:
        atoms, n_anions, net_charge, contact, pair, cn = build_system(key, seed=args.seed)
        fname = f"La3+_{key}_droplet.xyz"
        path = os.path.join(args.output_dir, fname)
        write(path, atoms)

        q = f"net {net_charge:+d}" if net_charge != 0 else "neutral"
        print(f"  {fname:<28s}  {len(atoms):>4d} atoms  "
              f"(1 La³⁺ + {n_anions} {key:<3s} + {N_WATER} H₂O)  {q:<8s}"
              f"  CN {cn[0]} @ {cn[1]:.2f} Å  min contact {contact:.2f} Å ({pair})")

    print(f"\nDone — {len(ANIONS)} files in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
