import torch

from .encoder import Encoder
from .decoder import Decoder
from .seq2seq import Seq2Seq

from .dataset import (
    english_vocab,
    french_vocab,
    encode_sentence
)

from .config import *
from common.utils import load_model


# Reverse French Vocabulary
# -----------------------------

idx_to_word = {
    idx: word
    for word, idx in french_vocab.items()
}


# Build Model
# -----------------------------

INPUT_DIM = len(english_vocab)
OUTPUT_DIM = len(french_vocab)

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

load_model(
    model,
    MODEL_PATH,
    DEVICE
)


# Translate Function
# -----------------------------

def translate(sentence):

    model.eval()

    encoded = encode_sentence(
        sentence,
        english_vocab
    )

    src = torch.tensor(
        encoded,
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hidden, cell = model.encoder(src)

    x = torch.tensor(
        [SOS_IDX],
        device=DEVICE
    )

    translated_sentence = []

    for _ in range(20):

        with torch.no_grad():

            prediction, hidden, cell = model.decoder(
                x,
                hidden,
                cell
            )

        best_guess = prediction.argmax(1)

        token = best_guess.item()

        if token == EOS_IDX:
            break

        translated_sentence.append(
            idx_to_word[token]
        )

        x = best_guess

    return " ".join(translated_sentence)


# Test
# -----------------------------

if __name__ == "__main__":

    sentence = "i love ai"

    translation = translate(sentence)

    print("English :", sentence)
    print("French  :", translation)