import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        device
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        src,
        target,
        teacher_forcing_ratio=0.5
    ):

        # src shape:
        # (batch_size, src_length)

        # target shape:
        # (batch_size, target_length)

        batch_size = target.shape[0]

        target_length = target.shape[1]

        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(
            batch_size,
            target_length,
            target_vocab_size
        ).to(self.device)

        # Encoder

        hidden, cell = self.encoder(src)

        # First decoder input = <START>

        x = target[:, 0]

        for t in range(1, target_length):

            prediction, hidden, cell = self.decoder(
                x,
                hidden,
                cell
            )

            outputs[:, t] = prediction

            best_guess = prediction.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio

            x = target[:, t] if teacher_force else best_guess

        return outputs