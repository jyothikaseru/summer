import torch
import torch.nn as nn
from .attention import Attention

class Decoder(nn.Module):

    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        # Convert token IDs to dense vectors
        self.embedding = nn.Embedding(
            output_dim,
            embedding_dim
        )

        # LSTM Decoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim+hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.attention = Attention()
        # Final layer to predict next word
        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )

    def forward(
        self,
        x,
        hidden,
        encoder_outputs,
        cell
    ):

        # x shape:
        # (batch_size)

        x = x.unsqueeze(1)

        # x shape:
        # (batch_size, 1)

        embedded = self.embedding(x)

        # embedded shape:
        # (batch_size, 1, embedding_dim)

        context, attentionweight = self.attention(encoder_outputs, hidden[-1])

        # context shape:
        # (batch_size, hidden_dim)

        attended_context = torch.cat([embedded, context.unsqueeze(1)], dim=2    )
        # attended_context shape:
        # (batch_size, 1, embedding_dim + hidden_dim)

        output,(hidden, cell) = self.lstm(
            attended_context,
            (hidden, cell)
        )

        # output shape:
        # (batch_size, 1, hidden_dim)

        prediction = self.fc(
            output.squeeze(1)
        )

        # prediction shape:
        # (batch_size, output_dim)

        return prediction, hidden, cell, attentionweight