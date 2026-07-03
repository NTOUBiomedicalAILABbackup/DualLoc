"""
config.py
=========
Centralized configuration for hyperparameters, paths, and device settings
for the ProtT5-based multi-label protein subcellular localization model.
"""

import os
import torch

# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_DIR = "/home/kchang/T5"
DATA_FILE = "protein_multi_data.tsv"
CHECKPOINT_NAME = "checkpoint_base_ProtT5_Dynamic_Label_Predict.pth"
BEST_MODEL_NAME = "MLTC_model_state_base_ProtT5_Dynamic_Label_Predict.bin"

CHECKPOINT_PATH = os.path.join(DATA_DIR, CHECKPOINT_NAME)
BEST_MODEL_PATH = os.path.join(DATA_DIR, BEST_MODEL_NAME)

# ------------------------------------------------------------------
# Pretrained backbone
# ------------------------------------------------------------------
PRETRAINED_MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
MAX_LEN = 512
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 2e-5          # Learning rate actually used by the AdamW optimizer
THRESHOLD = 0.5                 # Sigmoid decision threshold for classification
N_SAMPLES = 28303                # Number of rows to subsample from the raw dataset
RANDOM_STATE = 77
TEST_SIZE = 0.30

# ------------------------------------------------------------------
# Label definitions (10 subcellular localization categories)
# ------------------------------------------------------------------
LABEL_COLUMNS = [
    "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion",
    "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus",
    "Peroxisome",
]

# Number of repeated evaluation runs (for ensemble / robustness assessment)
NUM_EVAL_RUNS = 5
