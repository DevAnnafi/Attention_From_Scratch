import torch 
import torch.nn as nn
from src.layers import EncoderLayer

def test_encoder_layer_shape():
    B = 2
    S = 6
    d_model = 8
    num_heads = 4
    d_ff = 5
    x = torch.randn(B, S, d_model)
    el = EncoderLayer(d_model, num_heads, d_ff)
    out = el(x)
    assert out.shape == (B, S, d_model)