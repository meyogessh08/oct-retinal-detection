"""Central configuration for the OCT retinal disease project.

Everything you might want to tweak lives here, so the other modules stay clean.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"

# ---------------------------------------------------------------- dataset
# Hugging Face dataset id. Replace with the exact id used in the course notebook
# ("OCT dataset.ipynb") if it differs.
HF_DATASET_ID = "Falah/Retinal_OCT2017"

# Fallback label names. ALWAYS confirm with:
#   dataset["train"].features["label"].names
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------- subset sizes
TRAIN_PER_CLASS = 4000   # balanced training subset
VAL_PER_CLASS = 500      # carved from the remaining train images
SEED = 42

# ---------------------------------------------------------------- image / training
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2          # set to 0 on Windows if DataLoader hangs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Models supported by src/models.py
MODEL_NAMES = ["custom_cnn", "resnet18", "vgg16", "mobilenet_v2", "densenet121"]
