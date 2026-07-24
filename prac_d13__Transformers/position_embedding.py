import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length):
        super().__init__()

        # Create position indices: (max_seq_length, 1)
        position = torch.arange(max_seq_length).unsqueeze(1).float()

        # Compute the scaling factor for the sine and cosine functions 
        div_term = torch.exp(
            # array of shape (embed_dim/2,) with values [0, 2, 4, ..., embed_dim-2] multiplied by -log(10000)/embed_dim
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, embed_dim)

        # Apply sine to even indices

        #           dim0        dim1        dim2        dim3        dim4        dim5 ...
        #   pos0     sin(...)    cos(...)    sin(...)    cos(...)    sin(...)    cos(...)
        #   pos1     sin(...)    cos(...)    sin(...)    cos(...)    sin(...)    cos(...)
        #   pos2     sin(...)    cos(...)    sin(...)    cos(...)    sin(...)    cos(...)
        #   pos3     sin(...)    cos(...)    sin(...)    cos(...)    sin(...)    cos(...)


        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension -> (1, max_seq_length, embed_dim)
        pe = pe.unsqueeze(0)

        # Store as a non-trainable tensor
        self.register_buffer("pe", pe)

    def forward(self, x):
    
        # x: (batch_size, seq_len, embed_dim)
    
        seq_len = x.size(1)

        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]

        return x