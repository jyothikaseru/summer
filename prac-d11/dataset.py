# dataset.py

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import PAD_IDX

# Toy English-French sentence pairs

sentence_pairs = [
    ("i am happy", "je suis heureux"),
    ("i am sad", "je suis triste"),
    ("i love ai", "j'aime l'ia"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("how are you", "comment allez-vous"),
    ("i am hungry", "j'ai faim"),
    ("see you later", "à plus tard"),
    ("what is your name", "comment vous appelez-vous"),
    ("my name is john", "je m'appelle john"),
    ("i like coffee", "j'aime le café"),
    ("i like tea", "j'aime le thé"),
    ("where are you", "où êtes-vous"),
    ("i am from india", "je viens d'inde"),
    ("welcome", "bienvenue"),
    ("goodbye", "au revoir"),
    ("yes", "oui"),
    ("no", "non"),
    ("please", "s'il vous plaît")
]

print("Number of sentence pairs:", len(sentence_pairs))

print("\nFirst sentence pair:")
print(sentence_pairs[0])

print("\nEnglish:")
print(sentence_pairs[0][0])

print("\nFrench:")
print(sentence_pairs[0][1])


print("\nAll sentence pairs:\n")

for english, french in sentence_pairs:
    print(f"English: {english}")
    print(f"French : {french}")
    print("-" * 40)

# create vocabularies for English and French
# --------------------------------------------
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


english_vocab = {
    PAD_TOKEN: 0,
    SOS_TOKEN: 1,
    EOS_TOKEN: 2,
    UNK_TOKEN: 3
}

french_vocab = {
    PAD_TOKEN: 0,
    SOS_TOKEN: 1,
    EOS_TOKEN: 2,
    UNK_TOKEN: 3
}

for english, french in sentence_pairs:
    for word in english.split():
        if word not in english_vocab:
            english_vocab[word] = len(english_vocab)

    for word in french.split():
        if word not in french_vocab:
            french_vocab[word] = len(french_vocab)

print("\nEnglish Vocabulary\n")

for word, idx in english_vocab.items():
    print(f"{word:15} -> {idx}")

print("\nFrench Vocabulary\n")

for word, idx in french_vocab.items():
    print(f"{word:15} -> {idx}")


print("\nEnglish Vocabulary Size:", len(english_vocab))
print("French Vocabulary Size :", len(french_vocab))

# converting sentences to numerical representations (tokenization)

def encode_sentence(sentence, vocabulary):

    tokens = sentence.split()

    encoded = [vocabulary[SOS_TOKEN]]

    for token in tokens:

        if token in vocabulary:
            encoded.append(vocabulary[token])
        else:
            encoded.append(vocabulary[UNK_TOKEN])

    encoded.append(vocabulary[EOS_TOKEN])

    return encoded


english = sentence_pairs[0][0]
french = sentence_pairs[0][1]

english_ids = encode_sentence(
    english,
    english_vocab
)

french_ids = encode_sentence(
    french,
    french_vocab
)

print("English:", english)
print("Encoded:", english_ids)

print()

print("French:", french)
print("Encoded:", french_ids)


encoded_english = []

encoded_french = []

for english, french in sentence_pairs:

    encoded_english.append(
        encode_sentence(
            english,
            english_vocab
        )
    )

    encoded_french.append(
        encode_sentence(
            french,
            french_vocab
        )
    )


for i in range(len(encoded_english)):

    print("English IDs:", encoded_english[i])

    print("French IDs :", encoded_french[i])

    print("-"*50)

# building a custom dataset class for translation
class TranslationDataset(Dataset):

    def __init__(
        self,
        source_sentences,
        target_sentences
    ):

        self.source = source_sentences
        self.target = target_sentences

    def __len__(self):

        return len(self.source)

    def __getitem__(self, idx):

        src = torch.tensor(
            self.source[idx],
            dtype=torch.long
        )

        tgt = torch.tensor(
            self.target[idx],
            dtype=torch.long
        )

        return src, tgt

dataset = TranslationDataset(
    encoded_english,
    encoded_french
)
print("Dataset Size:", len(dataset))

src, tgt = dataset[0]

print("Source Tensor:")
print(src)

print()

print("Target Tensor:")
print(tgt)

# collate function for DataLoader to handle variable-length sequences
def collate_fn(batch):

    src_batch = []
    tgt_batch = []

    for src, tgt in batch:

        src_batch.append(src)
        tgt_batch.append(tgt)

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=PAD_IDX
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=PAD_IDX
    )

    return src_batch, tgt_batch

# batch = [

# (torch.tensor([1,4,5,2]),
#  torch.tensor([1,9,10,2])),

# (torch.tensor([1,7,8,9,2]),
#  torch.tensor([1,11,12,13,2])),

# (torch.tensor([1,15,16,17,18,2]),
#  torch.tensor([1,20,21,22,23,2]))
# ]

# src_batch = [x[0] for x in batch]
# tgt_batch = [x[1] for x in batch]

# src_batch = pad_sequence(src_batch,
#                          batch_first=True,
#                          padding_value=PAD_IDX)

# tgt_batch = pad_sequence(tgt_batch,
#                          batch_first=True,
#                          padding_value=PAD_IDX)
# print("Source Batch:")
# print(src_batch)

# print("\nTarget Batch:")
# print(tgt_batch)


# creating a DataLoader for batching and shuffling the data
train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

for src, tgt in train_loader:

    print("Source Batch:")
    print(src)
    print("Shape:", src.shape)
    print()

    print("Target Batch:")
    print(tgt)
    print("Shape:", tgt.shape)
    
print(len(train_loader))  


