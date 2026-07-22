import torch
import torch.nn as nn
import torch.optim as optim

from .encoder import Encoder
from .decoder import Decoder
from .seq2seq import Seq2Seq

from .dataset import (
    train_loader,
    english_vocab,
    french_vocab
)

from .config import *

from common.utils import (
    save_model,
    count_parameters
)


# -----------------------------
# Vocabulary Sizes
# -----------------------------

INPUT_DIM = len(english_vocab)
OUTPUT_DIM = len(french_vocab)


# -----------------------------
# Build Model
# -----------------------------

encoder = Encoder(
    input_dim=INPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

decoder = Decoder(
    output_dim=OUTPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

model = Seq2Seq(
    encoder,
    decoder,
    DEVICE
).to(DEVICE)


print(f"Trainable Parameters: {count_parameters(model):,}")


# -----------------------------
# Loss & Optimizer
# -----------------------------
# ignore the padding index in the loss calculation because we don't want to penalize the model for predicting padding tokens.
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training Loop
# -----------------------------

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for src, tgt in train_loader:

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        optimizer.zero_grad()

        output = model(
            src,
            tgt,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )

        output = output[:, 1:].reshape(
            -1,
            OUTPUT_DIM
        )

        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(
            output,
            tgt
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss/len(train_loader):.4f}"
    )


# -----------------------------
# Save Model
# -----------------------------

save_model(
    model,
    MODEL_PATH
)