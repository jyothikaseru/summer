import torch
import torch.nn as nn
from .config import *
from .position_embedding import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Token Embedding
        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBED_DIM
        )

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            embed_dim=EMBED_DIM,
            max_seq_length=MAX_SEQ_LEN
        )

        # Single Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NHEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True
        )

        # Stack Multiple Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_ENCODER_LAYERS
        )

        # Output Layer
        self.fc = nn.Linear(
            EMBED_DIM,
            VOCAB_SIZE
        )

    def forward(self, x, src_key_padding_mask=None):
    
        # x: (batch_size, seq_len)
        

        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.embedding(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer Encoder
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        # Project to vocabulary size
        x = self.fc(x)

        return x