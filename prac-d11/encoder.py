import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        # Convert token IDs to dense vectors
        self.embedding = nn.Embedding(
            input_dim,
            embedding_dim
        )

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, src):

        # src shape:
        # (batch_size, sequence_length)

        embedded = self.embedding(src)

        # embedded shape:
        # (batch_size, sequence_length, embedding_dim)

        outputs, (hidden, cell) = self.lstm(embedded)

        # outputs shape:
        # (batch_size, sequence_length, hidden_dim)

        # hidden shape:
        # (num_layers, batch_size, hidden_dim)

        # cell shape:
        # (num_layers, batch_size, hidden_dim)

        return hidden, cell