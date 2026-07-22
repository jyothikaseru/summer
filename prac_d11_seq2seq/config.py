import torch

# ==========================
# Device
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# Vocabulary
# ==========================

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ==========================
# Model Hyperparameters
# ==========================

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1


# ==========================
# Training Hyperparameters
# ==========================

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 100

TEACHER_FORCING_RATIO = 0.5


# ==========================
# Paths
# ==========================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

MODEL_PATH = CHECKPOINT_DIR / "seq2seq_best_model.pth"