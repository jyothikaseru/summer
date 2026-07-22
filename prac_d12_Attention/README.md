# Day 12 – Attention Mechanism for Neural Machine Translation

## Overview

This project extends the vanilla Seq2Seq model by incorporating a **Dot-Product Attention Mechanism**. Instead of relying only on the encoder's final hidden state, the decoder dynamically focuses on different encoder outputs while generating each target word.

The project translates simple English sentences into French using a toy parallel dataset built with PyTorch.

---

## Project Structure

```
prac_d12_Attention/
│
├── attention.py              # Dot-Product Attention module
├── encoder.py                # LSTM Encoder
├── decoder.py                # Attention-based LSTM Decoder
├── seq2seq_attention.py      # Complete Seq2Seq model
├── train.py                  # Training script
├── evaluate.py               # Translation + Attention visualization
├── toy_dataset.py            # Dataset and vocabulary
├── config.py                 # Hyperparameters
├── checkpoints/
│   └── attention_best_model.pth
├── plots/
│   └── attention_heatmap.png 
└── README.md
```

---

## Features

- LSTM Encoder
- Dot-Product Attention
- Attention-based LSTM Decoder
- Teacher Forcing
- CrossEntropy Loss
- Adam Optimizer
- Model Checkpoint Saving
- Translation Inference
- Attention Heatmap Visualization
- Comparison with Vanilla Seq2Seq Model

---

## Attention Mechanism

For every generated target word, the decoder computes attention scores over all encoder outputs.

### Step 1: Compute Attention Scores

```
score = Encoder Outputs × Decoder Hidden State
```

### Step 2: Convert Scores to Probabilities

```
Attention Weights = Softmax(Score)
```

### Step 3: Compute Context Vector

```
Context = Weighted Sum of Encoder Outputs
```

The context vector is concatenated with the decoder input embedding before being passed into the LSTM.

---

## Model Architecture

```
Input Sentence
      │
      ▼
Embedding
      │
      ▼
Encoder LSTM
      │
      ├─────────────── Encoder Outputs ───────────────┐
      │                                               │
Final Hidden State                                    │
      │                                               │
      ▼                                               │
Attention Mechanism ◄──────────────────────────────────┘
      │
Context Vector
      │
      ▼
Concatenate(Context, Embedding)
      │
      ▼
Decoder LSTM
      │
      ▼
Linear Layer
      │
      ▼
Predicted French Word
```

---

## Training

Train the model using:

```bash
python -m prac_d12_Attention.train
```

The best model is automatically saved inside:

```
checkpoints/
```

---

## Evaluation

Run:

```bash
python -m prac_d12_Attention.evaluate
```

This performs:

- English → French translation
- Attention heatmap generation
- Comparison with the vanilla Seq2Seq model

---

## Example

**Input**

```
i love ai
```

**Output**

```
j'aime l'ia
```

---

## Attention Heatmap

The attention matrix visualizes how much the decoder attends to each source word while generating every target word.

Example:

```
            Source Sentence

        i   love   ai

j'aime   ██░░░
l'ia     ░███░
```

The heatmap helps interpret the model's translation process by showing the alignment between source and target words.

---

## Technologies Used

- Python
- PyTorch
- Matplotlib
- NumPy

---

## Learning Outcomes

After completing this project, I understood:

- Why vanilla Seq2Seq struggles with long sentences
- The intuition behind the Attention Mechanism
- Dot-Product Attention
- Context Vector computation
- Attention score calculation
- Teacher Forcing
- Encoder–Decoder architecture
- Translation inference
- Attention visualization using heatmaps

---

## Future Improvements

- Bahdanau Attention
- Luong Attention
- Beam Search Decoding
- BLEU Score Evaluation
- Larger Translation Dataset
- Transformer Architecture

---

## Author

**Jyothika Seru**
