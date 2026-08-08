import torch 
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        return self.embedding(x) * self.scale

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        S = x.shape[1]
        return x + self.pe[:S]

pe_module = PositionalEncoding(d_model=64, max_len=100)
plt.imshow(pe_module.pe.numpy(), aspect='auto', cmap='RdBu')
plt.colorbar()
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding')
plt.show()

        


