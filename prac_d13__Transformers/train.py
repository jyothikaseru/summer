import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .transformer_model import TransformerModel
from .config import *


device = DEVICE

# Dummy Dataset

num_samples = 1000

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

dataset = TensorDataset(inputs, targets)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)



model = TransformerModel().to(device)

print("Model created")

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training Loop


for epoch in range(NUM_EPOCHS):

    

    model.train()

    total_loss = 0

    for batch_inputs, batch_targets in train_loader:

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()

        outputs = model(batch_inputs)

        loss = criterion(
            outputs.view(-1, VOCAB_SIZE),
            batch_targets.view(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")


# Save Model


torch.save(model.state_dict(), MODEL_PATH)

print(f"\nModel saved to {MODEL_PATH}")