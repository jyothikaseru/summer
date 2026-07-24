import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .transformer_model import TransformerModel
from .config import *


device = DEVICE

# Dummy Test Dataset


num_samples = 200

inputs = torch.randint(
    low=0,
    high=VOCAB_SIZE,
    size=(num_samples, MAX_SEQ_LEN)
)

targets = torch.randint(
    low=0,
    high=VOCAB_SIZE,
    size=(num_samples, MAX_SEQ_LEN)
)

test_dataset = TensorDataset(inputs, targets)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Load Model


model = TransformerModel().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()


criterion = nn.CrossEntropyLoss()


# Evaluation


total_loss = 0

with torch.no_grad():

    for batch_inputs, batch_targets in test_loader:

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        outputs = model(batch_inputs)

        loss = criterion(
            outputs.view(-1, VOCAB_SIZE),
            batch_targets.view(-1)
        )

        total_loss += loss.item()

avg_loss = total_loss / len(test_loader)

print(f"Test Loss : {avg_loss:.4f}")