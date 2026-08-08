import torch
import torch.nn as nn
from src.embedding import TokenEmbedding, PositionalEncoding

def test_token_embedding_shape():
    B = 2
    S = 6
    d_model = 8
    vocab_size = 100
    x = torch.randint(0, vocab_size, (B,S))
    te = TokenEmbedding(vocab_size, d_model)
    out = te(x)
    assert out.shape == (B, S, d_model)

def test_positional_encoding():
    B = 2
    S = 6
    d_model = 8
    max_len = 100
    x = torch.randn(B, S, d_model)
    pe = PositionalEncoding(d_model, max_len)
    out = pe(x)
    assert out.shape == (B, S, d_model)
