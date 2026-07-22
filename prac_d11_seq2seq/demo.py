import torch

from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
INPUT_DIM = 1000
OUTPUT_DIM = 1000
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1

BATCH_SIZE = 32
SRC_LEN = 10
TARGET_LEN = 12


# Create Encoder
encoder = Encoder(
    input_dim=INPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

# Create Decoder
decoder = Decoder(
    output_dim=OUTPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

# Create Seq2Seq Model
model = Seq2Seq(
    encoder,
    decoder,
    device
).to(device)


# Dummy Input
src = torch.randint(
    0,
    INPUT_DIM,
    (BATCH_SIZE, SRC_LEN)
).to(device)

target = torch.randint(
    0,
    OUTPUT_DIM,
    (BATCH_SIZE, TARGET_LEN)
).to(device)


# Forward Pass
outputs = model(src, target)


print("Source Shape :", src.shape)
print("Target Shape :", target.shape)
print("Output Shape :", outputs.shape)