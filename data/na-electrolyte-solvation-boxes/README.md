---
license: cc-by-4.0
task_categories:
  - other
tags:
  - chemistry
  - molecular-dynamics
  - electrolyte
  - sodium-ion-battery
  - solvation
  - simulation
  - MLIP
pretty_name: Na-Ion Electrolyte Solvation Boxes
---

# Na-Ion Electrolyte Solvation Boxes

Periodic simulation boxes of sodium-ion battery electrolytes in extended XYZ format, equilibrated with machine-learned interatomic potentials (MLIPs). Covers 7 sodium salts, 21 solvents/cosolvents, and concentrations from dilute (0.1 M) through water-in-salt (21 M), including carbonate, ether, glyme, ionic liquid, phosphate, and aqueous systems.

## Subsets

| Subset | Boxes | Protocol | Potential | Status |
|--------|-------|----------|-----------|--------|
| `NVT_OrbMolV2/` | 79 | NVT Langevin (50 ps, 300 K) | OrbMol-v2 (Orbital Materials) | Complete |
| `NVT_OrbMolV2/npt_extended/` | 4 | NVT (50 ps) then NPT (100 ps, 300 K, 1 atm) | OrbMol-v2 (Orbital Materials) | Complete |
| `NPT_FAIRChem-UMA/` | 83 | NPT (300 K, 1 atm) | UMA-s-1.2 (FAIRChem/Meta) | In progress |
| `molecules/` | 32 | n/a | n/a | Input geometries |

### NVT_OrbMolV2

> Initial configurations were packed with Packmol and geometry-optimized with the FIRE algorithm. Systems were then thermalized under NVT Langevin dynamics (50 ps, 300 K) using the OrbMol-v2 potential (Orbital Materials), trained on the OMol25 dataset.

79 equilibrated boxes. NVT does not adjust box volume, so densities reflect the Packmol packing estimate, not the true equilibrium density.

### NVT_OrbMolV2/npt_extended

> A subset of 4 boxes that were further equilibrated under NPT dynamics (100 ps, 300 K, 1 atm) using OrbMol-v2 to relax the box volume to equilibrium density.

These boxes went through the full NVT then NPT pipeline with a single consistent potential (OrbMol-v2).

### NPT_FAIRChem-UMA

> Initial configurations were packed with Packmol and geometry-optimized with the FIRE algorithm. Systems were then equilibrated under NPT at 300 K and 1 atm using the OMol25-trained UMA potential (FAIRChem).

83 boxes (the same 79 compositions as the NVT set, plus 4 that failed NVT with OrbMol). These start from FIRE-optimized Packmol configurations directly (no prior NVT) and use NPT to simultaneously thermalize and relax the density. Still in progress, completed boxes get added as they finish.

### molecules

32 individual molecule PDB files (ions, solvents, ionic liquid components) used as Packmol inputs. Geometries were generated with RDKit ETKDG + MMFF94 optimization; CTFSI and TTE were corrected with xTB GFN2 optimization and verified against reference bond lengths.

## Construction Pipeline

1. 3D geometry construction in Avogadro with Universal Force Field (UFF) optimization, exported as PDB (xTB GFN2 correction applied to CTFSI and TTE)
2. Packing with Packmol, 2.0 Å tolerance, cubic periodic box sized from the target concentration
3. Geometry optimization with FIRE (200 steps) using the OrbMol potential
4. Equilibration under NVT and/or NPT molecular dynamics with an MLIP (see subset descriptions)

## Electrolyte Coverage

### Salts
NaPF₆, NaClO₄, NaFSI, NaTFSI, NaOTf, NaBF₄, NaDFOB, NaCTFSI, KFSI

### Solvents
DME, diglyme, triglyme, tetraglyme, EC, PC, DMC, DEC, EMC, FEC, TMP, TEP, BTFE, TTE, ethyl acetate, methyl propionate, EGDEE, CPME, water

### Ionic Liquids
Pyr13FSI, Pyr14TFSI, EMImFSI

### Categories
- Carbonate electrolytes (EC/PC/DMC/DEC blends)
- Dilute ether and glyme electrolytes
- High-concentration ether electrolytes (up to 4 M)
- Localized high-concentration electrolytes (LHCE with BTFE/TTE diluents)
- Ionic liquid electrolytes
- Flame-retardant phosphate electrolytes
- Weakly solvating and low-temperature electrolytes
- Aqueous water-in-salt electrolytes (up to 21 M)
- Concentration series (NaPF₆ in diglyme, NaFSI in DME, NaClO₄ in water)
- Anion series (5 anions in DME at 1 M)
- Glyme chain-length series (DME through tetraglyme)
- Cosolvent ratio series (EC:DEC from 1:9 to 9:1)
- LHCE diluent series (varying BTFE content)

## Contents: NVT_OrbMolV2 (79 boxes)

| File | Electrolyte |
|------|------------|
| `NaBF4-NaPF6_EC-PC-1-1_0p5M-0p5M.xyz` | 0.5 M NaBF4 + 0.5 M NaPF6 in EC:PC 1:1 |
| `NaBF4_tetraglyme-TEP-3-7_1M.xyz` | 1 M NaBF4 in tetraglyme:TEP 3:7 |
| `NaCTFSI-NaFSI_water_14p1M-31p5M.xyz` | 14.1 M NaCTFSI + 31.5 M NaFSI in water |
| `NaClO4_DME_1M.xyz` | 1 M NaClO4 in DME |
| `NaClO4_DME_1M-r2.xyz` | 1 M NaClO4 in DME (rep 2) |
| `NaClO4_EC-DEC-1-1_1M.xyz` | 1 M NaClO4 in EC:DEC 1:1 |
| `NaClO4_EC-DMC-1-1_1M.xyz` | 1 M NaClO4 in EC:DMC 1:1 |
| `NaClO4_EC-DME-1-1_1M.xyz` | 1 M NaClO4 in EC:DME 1:1 |
| `NaClO4_EC-PC-1-1_1M.xyz` | 1 M NaClO4 in EC:PC 1:1 |
| `NaClO4_PC_1M.xyz` | 1 M NaClO4 in PC |
| `NaClO4_water_1M.xyz` | 1 M NaClO4 in water |
| `NaClO4_water_5M.xyz` | 5 M NaClO4 in water |
| `NaClO4_water_10M.xyz` | 10 M NaClO4 in water |
| `NaClO4_water_17M.xyz` | 17 M NaClO4 in water |
| `NaClO4_water_17M-r2.xyz` | 17 M NaClO4 in water (rep 2) |
| `NaFSI_diglyme_1M.xyz` | 1 M NaFSI in diglyme |
| `NaFSI_diglyme-BTFE-1-2_1p2M.xyz` | 1.2 M NaFSI in diglyme:BTFE 1:2 |
| `NaFSI_DME_0p5M.xyz` | 0.5 M NaFSI in DME |
| `NaFSI_DME_1M.xyz` | 1 M NaFSI in DME |
| `NaFSI_DME_1M-r2.xyz` | 1 M NaFSI in DME (rep 2) |
| `NaFSI_DME_2M.xyz` | 2 M NaFSI in DME |
| `NaFSI_DME_3M.xyz` | 3 M NaFSI in DME |
| `NaFSI_DME_3p8M.xyz` | 3.8 M NaFSI in DME |
| `NaFSI_DME_4M.xyz` | 4 M NaFSI in DME |
| `NaFSI_DME_4M-r2.xyz` | 4 M NaFSI in DME (rep 2) |
| `NaFSI_DME_5M.xyz` | 5 M NaFSI in DME |
| `NaFSI_DME_5M-r2.xyz` | 5 M NaFSI in DME (rep 2) |
| `NaFSI_DME_r-1-1p2.xyz` | NaFSI:DME 1:1.2 mol ratio |
| `NaFSI_DME-BTFE-1-2_2p1M.xyz` | 2.1 M NaFSI in DME:BTFE 1:2 |
| `NaFSI_DME-BTFE_r-1-1p2-1.xyz` | NaFSI:DME:BTFE 1:1.2:1 mol ratio |
| `NaFSI_DME-BTFE_r-1-1p2-2.xyz` | NaFSI:DME:BTFE 1:1.2:2 mol ratio |
| `NaFSI_DME-BTFE_r-1-1p2-3.xyz` | NaFSI:DME:BTFE 1:1.2:3 mol ratio |
| `NaFSI_DME-TTE_r-1-2p25-3.xyz` | NaFSI:DME:TTE 1:2.25:3 mol ratio |
| `NaFSI-KFSI_DME_1M-0p05M.xyz` | 1 M NaFSI + 0.05 M KFSI in DME |
| `NaFSI_Pyr13FSI_3p8M.xyz` | 3.8 M NaFSI in Pyr13FSI |
| `NaFSI_Pyr13FSI_r-1-9.xyz` | NaFSI:Pyr13FSI 1:9 mol ratio |
| `NaFSI_TMP_3p3M.xyz` | 3.3 M NaFSI in TMP |
| `NaFSI_water_21M.xyz` | 21 M NaFSI in water |
| `NaOTf_diglyme_1M.xyz` | 1 M NaOTf in diglyme |
| `NaOTf_DME_1M.xyz` | 1 M NaOTf in DME |
| `NaOTf_DME_1M-r2.xyz` | 1 M NaOTf in DME (rep 2) |
| `NaOTf_tetraglyme_1M.xyz` | 1 M NaOTf in tetraglyme |
| `NaOTf_water_9p26M.xyz` | 9.26 M NaOTf in water |
| `NaPF6_diglyme_0p4M.xyz` | 0.4 M NaPF6 in diglyme |
| `NaPF6_diglyme_0p4M-r2.xyz` | 0.4 M NaPF6 in diglyme (rep 2) |
| `NaPF6_diglyme_0p6M.xyz` | 0.6 M NaPF6 in diglyme |
| `NaPF6_diglyme_0p8M.xyz` | 0.8 M NaPF6 in diglyme |
| `NaPF6_diglyme_1M.xyz` | 1 M NaPF6 in diglyme |
| `NaPF6_diglyme_1M-r2.xyz` | 1 M NaPF6 in diglyme (rep 2) |
| `NaPF6_diglyme_1M-r3.xyz` | 1 M NaPF6 in diglyme (rep 3) |
| `NaPF6_diglyme_2p5M.xyz` | 2.5 M NaPF6 in diglyme |
| `NaPF6_diglyme_2p5M-r2.xyz` | 2.5 M NaPF6 in diglyme (rep 2) |
| `NaPF6_diglyme-TMP-1-1_1M.xyz` | 1 M NaPF6 in diglyme:TMP 1:1 |
| `NaPF6_DME_1M.xyz` | 1 M NaPF6 in DME |
| `NaPF6_DME_1M-r2.xyz` | 1 M NaPF6 in DME (rep 2) |
| `NaPF6_DME_1M_FEC5wt.xyz` | 1 M NaPF6 in DME + 5 wt% FEC |
| `NaPF6_DME-TMP-7-3_1M.xyz` | 1 M NaPF6 in DME:TMP 7:3 |
| `NaPF6_EA_1M_FEC5wt.xyz` | 1 M NaPF6 in ethyl acetate + 5 wt% FEC |
| `NaPF6_EC-DEC-1-1_1M.xyz` | 1 M NaPF6 in EC:DEC 1:1 |
| `NaPF6_EC-DEC-1-9_1M.xyz` | 1 M NaPF6 in EC:DEC 1:9 |
| `NaPF6_EC-DEC-3-7_1M.xyz` | 1 M NaPF6 in EC:DEC 3:7 |
| `NaPF6_EC-DEC-3-7_1M_FEC2wt.xyz` | 1 M NaPF6 in EC:DEC 3:7 + 2 wt% FEC |
| `NaPF6_EC-DEC-7-3_1M.xyz` | 1 M NaPF6 in EC:DEC 7:3 |
| `NaPF6_EC-DEC-9-1_1M.xyz` | 1 M NaPF6 in EC:DEC 9:1 |
| `NaPF6_EC-DMC-1-1_1M.xyz` | 1 M NaPF6 in EC:DMC 1:1 |
| `NaPF6_EC-PC-1-1_1M.xyz` | 1 M NaPF6 in EC:PC 1:1 |
| `NaPF6_EGDEE-CPME-1-1_1M.xyz` | 1 M NaPF6 in EGDEE:CPME 1:1 |
| `NaPF6_EMC-FEC-1-1_0p6M.xyz` | 0.6 M NaPF6 in EMC:FEC 1:1 |
| `NaPF6_MP_1M_FEC5wt.xyz` | 1 M NaPF6 in methyl propionate + 5 wt% FEC |
| `NaPF6-NaDFOB_EC-DEC-3-7_1M-0p1M.xyz` | 1 M NaPF6 + 0.1 M NaDFOB in EC:DEC 3:7 |
| `NaPF6-NaFSI_DME_0p8M-0p2M.xyz` | 0.8 M NaPF6 + 0.2 M NaFSI in DME |
| `NaPF6_PC_1M.xyz` | 1 M NaPF6 in PC |
| `NaPF6_PC_1M_EMImFSI10molpct.xyz` | 1 M NaPF6 in PC + 10 mol% EMImFSI |
| `NaPF6_tetraglyme_1M.xyz` | 1 M NaPF6 in tetraglyme |
| `NaPF6_tetraglyme_1M-r2.xyz` | 1 M NaPF6 in tetraglyme (rep 2) |
| `NaPF6_triglyme_1M.xyz` | 1 M NaPF6 in triglyme |
| `NaTFSI_DME_1M.xyz` | 1 M NaTFSI in DME |
| `NaTFSI_EC-DMC-1-1_1M.xyz` | 1 M NaTFSI in EC:DMC 1:1 |
| `NaTFSI_PC_1M.xyz` | 1 M NaTFSI in PC |

**Total: 79 boxes**

## Naming Convention

Files follow the pattern: `salt_solvent_concentration.xyz`

### Concentration Units

- `M` is molar (mol/L of solution)
- `0p5M` is 0.5 M, the letter `p` stands in for the decimal point
- `r-1-2p25-3` is a molar ratio (salt:solvent:diluent), prefixed with `r`

### Salt Abbreviations

| Abbreviation | Formula | Full Name |
|---|---|---|
| NaPF6 | NaPF₆ | Sodium hexafluorophosphate |
| NaClO4 | NaClO₄ | Sodium perchlorate |
| NaFSI | NaN(SO₂F)₂ | Sodium bis(fluorosulfonyl)imide |
| NaTFSI | NaN(SO₂CF₃)₂ | Sodium bis(trifluoromethanesulfonyl)imide |
| NaOTf | NaCF₃SO₃ | Sodium trifluoromethanesulfonate (triflate) |
| NaBF4 | NaBF₄ | Sodium tetrafluoroborate |
| NaDFOB | NaBF₂(C₂O₄) | Sodium difluoro(oxalato)borate |
| NaCTFSI | NaN(CN)(SO₂CF₃) | Sodium cyano(trifluoromethanesulfonyl)imide |
| KFSI | KN(SO₂F)₂ | Potassium bis(fluorosulfonyl)imide |

### Solvent Abbreviations

| Abbreviation | Full Name |
|---|---|
| DME | 1,2-Dimethoxyethane (monoglyme) |
| diglyme | Diethylene glycol dimethyl ether |
| triglyme | Triethylene glycol dimethyl ether |
| tetraglyme | Tetraethylene glycol dimethyl ether |
| EC | Ethylene carbonate |
| PC | Propylene carbonate |
| DMC | Dimethyl carbonate |
| DEC | Diethyl carbonate |
| EMC | Ethyl methyl carbonate |
| FEC | Fluoroethylene carbonate |
| TMP | Trimethyl phosphate |
| TEP | Triethyl phosphate |
| BTFE | Bis(2,2,2-trifluoroethyl) ether |
| TTE | 1,1,2,2-Tetrafluoroethyl 2,2,3,3-tetrafluoropropyl ether |
| EA | Ethyl acetate |
| MP | Methyl propionate |
| EGDEE | Ethylene glycol diethyl ether (1,2-diethoxyethane) |
| CPME | Cyclopentyl methyl ether |
| Pyr13FSI | N-methyl-N-propylpyrrolidinium bis(fluorosulfonyl)imide |
| Pyr14TFSI | N-butyl-N-methylpyrrolidinium bis(trifluoromethanesulfonyl)imide |
| EMImFSI | 1-Ethyl-3-methylimidazolium bis(fluorosulfonyl)imide |

### Other Notation

| Notation | Meaning |
|---|---|
| `-r2`, `-r3` | Independent replicate (same composition, different random packing) |
| `FEC2wt`, `FEC5wt` | Additive at 2 or 5 weight percent |
| `EMImFSI10molpct` | Additive at 10 mol percent |
| `EC-DEC-3-7` | Mixed solvent with volume ratio (EC:DEC = 3:7) |
| `r-1-1p2-3` | Molar ratio specification (salt:solvent:diluent = 1:1.2:3) |

## Citation

If you use this dataset, please cite the OMol25 dataset and the respective MLIP models:

- OMol25: Levine et al., "The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models," 2025.
- OrbMol-v2: Orbital Materials, [orb-models](https://github.com/orbital-materials/orb-models)
- UMA: FAIRChem/Meta, [fairchem](https://github.com/FAIR-Chem/fairchem)
