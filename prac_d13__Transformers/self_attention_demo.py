import torch

batch_size = 2
seq_len = 5
embed_dim = 128

X = torch.randn(batch_size, seq_len, embed_dim) 


# Wq, Wk, and Wv are initialized as linear layers that project the input embeddings into query, key, and value spaces.
# why aren't they initialized as matrices because in PyTorch, the `torch.nn.Linear` layer is a convenient way to define a linear transformation that includes both the weight matrix and the bias term. It automatically handles the initialization of the weights and biases, making it easier to work with compared to manually defining weight matrices.
Wq = torch.nn.Linear(embed_dim, embed_dim)
Wk = torch.nn.Linear(embed_dim, embed_dim)
Wv = torch.nn.Linear(embed_dim, embed_dim)

Q = Wq(X)
K = Wk(X)
V = Wv(X)

scores = torch.matmul(Q, K.transpose(-2, -1))

scores = scores / (embed_dim ** 0.5)

weights = torch.softmax(scores, dim=-1)

context = torch.matmul(weights, V)

print("Input:", X.shape)
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)
print("Scores:", scores.shape)
print("Weights:", weights.shape)
print("Context:", context.shape)