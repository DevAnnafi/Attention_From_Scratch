import torch 
import torch.nn as nn
from src.model import Transformer

def test_transformer():
    B = 2
    S = 6
    d_model = 8
    num_heads = 4
    d_ff = 7
    N = 3
    vocab_size = 100
    max_len = 100
    src = torch.randint(0, vocab_size, (B, S))
    tgt = torch.randint(0, vocab_size, (B, S))
    t = Transformer(d_model, num_heads, d_ff, N, vocab_size, max_len)
    out = t(src, tgt)
    assert out.shape == (B, S, vocab_size)


