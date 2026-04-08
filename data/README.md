
## Datasets

This study utilizes three curated datasets derived from public protein databases,
all containing experimentally validated (ECO:0000269) nuclear-encoded eukaryotic
proteins with a minimum length of 40 amino acids.

**Swiss-Prot** (UniProt release 2021_03_23) serves as the primary training and
cross-validation set, comprising 28,303 unique sequences annotated across
10 subcellular compartments: cytoplasm, nucleus, extracellular space, cell membrane,
mitochondrion, plastid, endoplasmic reticulum, lysosome/vacuole, Golgi apparatus,
and peroxisome.

**Human Protein Atlas (HPA)** provides 1,717 non-redundant sequences for independent
external validation, covering 8 compartments (extracellular and plastid excluded).
Proteins sharing ≥30% global sequence identity with the Swiss-Prot training set
were excluded to prevent homology bias.

**Sorting Signal Dataset**, compiled from DeepLoc 2.0 and Swiss-Prot annotations,
supports classification of 9 sorting signal types: signal peptide (SP),
transmembrane segment (TM), mitochondrial targeting (MT), chloroplast transit (CH),
thylakoid transfer (TH), nuclear localization (NLS), nuclear export (NES),
peroxisomal targeting (PTS), and GPI-anchor (GPI) signals.

All three datasets were originally curated by [DeepLoc 2.0](https://services.healthtech.dtu.dk/services/DeepLoc-2.0/).



---

### Dataset 1 — Swiss-Prot Subcellular Localization (Training / Cross-Validation)

| Property        | Detail                                                                          |
|-----------------|---------------------------------------------------------------------------------|
| **Source**      | UniProt Swiss-Prot release `2021_03_23`                                         |
| **Curation**    | Adopted from DeepLoc 2.0                                                        |
| **Sequences**   | 28,303 unique non-redundant sequences                                           |
| **Compartments**| 10 (see full list below)                                                        |
| **Splits**      |5-fold cross-validation; 20% of training data used as validation set             |
| **Filter**      | ≥ 40 amino acids; excludes bacterial/archaeal sequences and protein fragments   |

**10 Subcellular Compartments:**

| # | Compartment          | Abbrev. |
|---|----------------------|---------|
| 1 | Cytoplasm            | CYT     |
| 2 | Nucleus              | NUC     |
| 3 | Extracellular        | EXT     |
| 4 | Cell membrane        | MEM     |
| 5 | Mitochondrion        | MIT     |
| 6 | Plastid              | PLA     |
| 7 | Endoplasmic Reticulum| ER      |
| 8 | Lysosome / Vacuole   | LYS     |
| 9 | Golgi apparatus      | GOL     |
|10 | Peroxisome           | PER     |



---

### Dataset 2 — Human Protein Atlas (Independent Validation)

| Property        | Detail                                               |
|-----------------|------------------------------------------------------|
| **Source**      | Human Protein Atlas (HPA)                            |
| **Curation**    | Adopted from DeepLoc 2.0                             |
| **Sequences**   | 1,717 non-redundant sequences                        |
| **Compartments**| 8 (Extracellular and Plastid **excluded**)           |
| **Confidence**  | Enhanced or Supported annotations only               |
| **Filter**      | Proteins with ≥ 30% global sequence identity to      |
|                 | Swiss-Prot training set are **excluded** (homology bias prevention) |

> This dataset is used **exclusively** for independent generalization testing.


---

### Dataset 3 — Sorting Signal Dataset (Training / Cross-Validation)

| Property        | Detail                                                                                                 |
|-----------------|--------------------------------------------------------------------------------------------------------|
| **Source**      | A high-confidence sorting signal dataset was compiled from annotations in DeepLoc 2.0 and Swiss-Prot   |
| **Signal Types**| 9 distinct sorting signal classes                                                                      |
| **Splits**      | 5-fold cross-validation; 20% of training data used as validation set                                   |

**9 Sorting Signal Classes:**

| Signal | Full Name                          | Primary Target Compartment |
|--------|------------------------------------|----------------------------|
| SP     | Signal Peptide                     | Extracellular (80.2%)      |
| TM     | Transmembrane Segment              | Cell membrane (69.9%)      |
| MT     | Mitochondrial Targeting Signal     | Mitochondrion (99.2%)      |
| CH     | Chloroplast Transit Peptide        | Plastid                    |
| TH     | Thylakoid Transfer Peptide         | Plastid (thylakoid lumen)  |
| NLS    | Nuclear Localization Signal        | Nucleus (100.0%)           |
| NES    | Nuclear Export Signal              | Nucleus / Cytoplasm        |
| PTS    | Peroxisomal Targeting Signal       | Peroxisome                 |
| GPI    | GPI-Anchor Signal                  | Cell membrane              |





