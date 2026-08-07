import pytest
import torch
from src.attention import split_heads, merge_heads

def test_split_heads_shape():
    B = 2
    S = 6
    d_model = 8
    H = 2
    x = torch.randn(B, S, d_model)
    result = split_heads(x, H)
    assert result.shape == (B, H, S, d_model // H)

def test_head_round_trip():
    B = 2
    S = 6
    d_model = 8
    H = 2
    x = torch.randn(B, S, d_model)
    split = split_heads(x, H)
    merge = merge_heads(split)
    assert torch.allclose(x, merge)
