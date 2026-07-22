import torch
import matplotlib.pyplot as plt

from .encoder import Encoder
from .decoder import Decoder
from .seq2seq_attention import Seq2Seq

from .toy_dataset import (
    english_vocab,
    french_vocab,
    encode_sentence
)

from .config import *
from common.utils import load_model

from prac_d11_seq2seq.evaluate import translate as translate_seq2seq


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
        encoder_outputs, hidden, cell = model.encoder(src)

    x = torch.tensor(
        [SOS_IDX],
        device=DEVICE
    )

    translated_sentence = []
    attention_matrix = []

    for _ in range(20):

        with torch.no_grad():

            prediction, hidden, cell, attention_weights = model.decoder(
                x,
                hidden,
                encoder_outputs,
                cell
            )
        # print(attention_weights)
        attention_matrix.append(
            attention_weights.squeeze(0).cpu()
        )

        best_guess = prediction.argmax(1)

        token = best_guess.item()

        if token == EOS_IDX:
            break

        translated_sentence.append(
            idx_to_word[token]
        )

        x = best_guess

    attention_matrix = torch.stack(attention_matrix)

    return " ".join(translated_sentence), attention_matrix


# Test
# -----------------------------

if __name__ == "__main__":

    sentence = "we develop advanced transformer models for natural language processing every day"

    translation, attention_matrix = translate(sentence)

    print("English :", sentence)
    print("French  :", translation)

    # Labels for heatmap
    source_words = ["<SOS>"] + sentence.split() + ["<EOS>"]
    target_words = translation.split() + ["<EOS>"]

    # Adjust labels if attention matrix size differs
    if attention_matrix.shape[1] != len(source_words):
        source_words = source_words[:attention_matrix.shape[1]]

    if attention_matrix.shape[0] != len(target_words):
        target_words = target_words[:attention_matrix.shape[0]]

    plt.figure(figsize=(8, 6))

    plt.imshow(
        attention_matrix.cpu().numpy(),
        cmap="viridis",
        aspect="auto"
    )

    plt.xticks(
        ticks=range(len(source_words)),
        labels=source_words,
        rotation=45
    )

    plt.yticks(
        ticks=range(len(target_words)),
        labels=target_words
    )

    plt.xlabel("Source Sentence")
    plt.ylabel("Generated Sentence")
    plt.title("Attention Heatmap")

    plt.colorbar(label="Attention Weight")

    plt.tight_layout()
    plt.show()

    test_sentences = [
        "i love ai",
        "i love machine learning",
        "she likes coffee",
        "he likes pizza",
        "we study deep learning",
        "they play football",
        "i am a student",
        "you are my friend",
        "she is reading a book",
        "we are learning artificial intelligence"
    ]

    print("=" * 90)
    print(f"{'English':35} {'Without Attention':30} {'With Attention'}")
    print("=" * 90)

    for sentence in test_sentences:

        translation1 = translate_seq2seq(sentence)
        translation2, _ = translate(sentence)

        print(f"{sentence:35} {translation1:30} {translation2}")