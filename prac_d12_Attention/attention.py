import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,encoder_outputs, decoder_hidden):
        # encoder_outputs shape:
        # (batch_size, seq_len, hidden_dim)
        # decoder_hidden shape:
        # (batch_size, hidden_dim)
        attentionscore=torch.bmm(encoder_outputs,decoder_hidden.unsqueeze(2)).squeeze(2)
        # attentionscore shape:
        # (batch_size, seq_len)
        attentionweight=torch.softmax(attentionscore,dim=1)
        # attentionweight shape:
        # (batch_size, seq_len)
        context=torch.matmul(attentionweight.unsqueeze(1),encoder_outputs).squeeze(1)
        # context shape:
        # (batch_size, hidden_dim)
        return context,attentionweight