"""
Configuration for Brain Tumor Classification with SAM.
Central place for paths, hyperparameters, and data settings.
"""
import os
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Expected data layout after download: data/Brain Tumor MRI Dataset/<class_name>/*.jpg
# Or: data/<class_name>/*.jpg depending on Kaggle extract structure
DATASET_SUBDIR = "Brain Tumor MRI Dataset"  # subfolder name inside data/ if present
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # folder names, order = label 0,1,2,3

# ---------- Data ----------
IMG_SIZE = 224
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
NUM_CLASSES = 4

# ---------- Preprocessing ----------
GAUSSIAN_BLUR_KERNEL = (5, 5)  # kernel size for Gaussian blur
GAUSSIAN_BLUR_SIGMA = (0.1, 2.0)  # sigma range
ROTATION_DEGREES = 10  # ± degrees for augmentation
BRIGHTNESS_FACTOR = (0.8, 1.2)  # range for brightness adjustment

# ---------- Training ----------
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ---------- Focal Loss ----------
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = None  # None = uniform; or list of 4 class weights

# ---------- SAM ----------
SAM_RHO = 0.05  # neighborhood radius ρ; try 0.05, 0.1, 0.2
USE_SAM = False  # set True in train_sam.py or via CLI

# ---------- Explainability ----------
IG_N_STEPS = 50  # path steps for Integrated Gradients
IG_BASELINE = "black"  # "black" or "blur"

# ---------- Reproducibility ----------
SEED = 42


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in (OUTPUTS_DIR, CHECKPOINTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_data_root():
    """Return path to dataset root (folder containing class subdirs)."""
    p = DATA_DIR / DATASET_SUBDIR
    if p.exists():
        return p
    if DATA_DIR.exists():
        # Maybe data is directly under data/
        for c in CLASS_NAMES:
            if (DATA_DIR / c).exists():
                return DATA_DIR
    return DATA_DIR
