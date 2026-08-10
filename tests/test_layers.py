import torch 
import torch.nn as nn
from src.layers import FeedForward, LayerNorm, EncoderLayer, Encoder

def test_feed_forward():
    B = 2
    S = 6
    d_model = 8
    d_ff = 5
    x = torch.randn(B, S, d_model)
    ff = FeedForward(d_model, d_ff)
    out = ff(x)
    assert out.shape == (B, S, d_model)

def test_layernorm_shape():
    B = 2
    S = 6
    d_model = 8
    x = torch.randn(B, S, d_model)
    ln = LayerNorm(d_model)
    out = ln(x)
    assert out.shape == (B, S, d_model)


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

def test_encoder_shape():
    B = 2
    S = 6
    d_model = 8
    num_heads = 4
    d_ff = 7
    N = 3
    x = torch.randn(B, S, d_model)
    e = Encoder(d_model, num_heads, d_ff, N)
    out = e(x)
    assert out.shape == (B, S, d_model)