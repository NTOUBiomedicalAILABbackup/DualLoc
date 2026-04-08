![DualLoc](Logo.png)


DualLoc is a multi-label protein subcellular localization predictor using full-parameter fine-tuning of dual Protein Language Models (PLMs). It predicts localization across **10 subcellular compartments** and **9 sorting signal types** simultaneously. 

📄 **Paper**: DualLoc: Full-parameter fine-tuning of dual transformer models for protein subcellular localization prediction

---

## Overview

DualLoc integrates two distinct PLM encoders trained simultaneously:
- **Encoder 1**: Initialized with pretrained weights (ProtBERT / ESM-2 / ProtT5) to leverage broad evolutionary context
- **Encoder 2**: Randomly initialized to specialize in task-specific localization patterns

Both encoders are augmented with multi-head attention, Gaussian smoothing, dropout layers, and fully connected classification layers. DualLoc also supports hierarchical prediction, where localization features feed into a secondary pathway for sorting signal classification.

---

## Quick Install

> We suggest installing all packages using conda (both Miniconda and [Anaconda](https://anaconda.org/) are supported).

### 1. First-time Setup

```bash
# Create conda environment
conda create -n DualLoc python=3.10 -y
conda activate DualLoc

# Install PyTorch with CUDA 11.8
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install required packages
pip install transformers sentencepiece protobuf
pip install scikit-learn pandas numpy matplotlib seaborn
pip install umap-learn tqdm

# Clone repository
git clone https://github.com/NTOUBiomedicalAILABbackup/DualLoc.git
cd DualLoc/
```

### 2. Subsequent Usage

```bash
conda activate DualLoc
```

---

## Usage

### Run the Complete Pipeline

```bash
python src/main.py
```

This executes the full pipeline:
1. Load and preprocess `signal_sorting.csv`
2. Split data using 5-fold cross-validation (20% of training data as validation set)
3. Initialize DualLoc-ProtT5 model
4. Train for 10 epochs with checkpoint saving
5. Evaluate and output metrics
6. Run single sequence prediction demo

### Single Sequence Prediction

Modify the `sample_text` variable in `main.py`:

```python
sample_text = "MYWSNQITRRLGERVQGFMSGISPQQMGE..."
classifier.predict_raw_text(trained_model, sample_text, target_list)
```

Output example:
```
Predicted localizations:
 - SP
 - TM
```

---

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




---


## Project Structure

```
DualLoc/
├── src/
│   ├── config.py       # Hyperparameters and path settings
│   ├── model.py        # Dual T5 encoder architecture
│   ├── dataset.py      # Tokenizer and DataLoader
│   ├── trainer.py      # Training loop with checkpointing
│   └── main.py         # Main pipeline entry point
├── data/
└── README.md
```

---

## Data Availability

The sorting signal dataset was compiled from annotations in DeepLoc 2.0 and Swiss-Prot, comprising 1,868 sequences across 9 sorting signal classes.

The subcellular localization training data is derived from manually reviewed Swiss-Prot entries (UniProt release 2021-03-23), containing 28,303 unique sequences across 10 compartments. Independent validation uses Human Protein Atlas (HPA) data with 1,717 non-redundant sequences.

> 🔒 The dataset files are not included in this repository. Please refer to [DeepLoc 2.0](https://services.healthtech.dtu.dk/services/DeepLoc-2.0/) to obtain and format the data according to the Input Data Format section above.

---

## Configuration

Key parameters in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Rostlab/prot_t5_xl_uniref50` | Pretrained PLM (auto-downloaded from HuggingFace) |
| `EPOCHS` | 20 | Number of training epochs |
| `LEARNING_RATE` | 2e-4 | AdamW learning rate |
| `MAX_LEN` | 512 | Maximum sequence length |
| `dropout_rate` | 0.3 | Dropout rate |
| `THRESHOLD` | 0.5 | Prediction threshold (multi-label) |
|num_labels_loc| 10 | Output classes for subcellular localization |
|num_labels_signal| 9 | Output classes for sorting signal classification |




