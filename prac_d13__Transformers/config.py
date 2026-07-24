import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = 1000

EMBED_DIM = 512

NHEADS = 8

FF_DIM = 2048

NUM_ENCODER_LAYERS = 6

NUM_DECODER_LAYERS = 6

MAX_SEQ_LEN = 20

DROPOUT = 0.1

LEARNING_RATE = 1e-4

BATCH_SIZE = 32

NUM_EPOCHS=5

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

MODEL_PATH = CHECKPOINT_DIR / "transformer_best_model.pth"