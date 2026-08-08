import pytest
import torch
from src.attention import attention, MultiHeadAttention

def test_attention():
    Q = torch.tensor([[3.0,4.0], [8.0,2.0]])
    K = torch.tensor([[7.0,6.0], [5.0,9.0]])
    V = torch.tensor([[1.0,2.0], [3.0,4.0]])
    result = (attention(Q,K,V))
    expected = torch.tensor([[2.9717, 3.9717], [1.0017, 2.0017]])
    assert torch.allclose(result, expected, atol=1e-4)

def test_multiheadattention():
    B = 2
    S = 6
    d_model = 8
    H = 2
    x = torch.randn(B, S, d_model)
    mha = MultiHeadAttention(8,2)
    out = mha(x,x,x)
    assert out.shape == (B, S, d_model)

def test_h1_equivalence():
    B = 2
    S = 6
    d_model = 8
    mha = MultiHeadAttention(d_model, 1)
    with torch.no_grad():
        mha.W_Q.weight.copy_(torch.eye(d_model))
        mha.W_Q.bias.copy_(torch.zeros(d_model))
        mha.W_K.weight.copy_(torch.eye(d_model))
        mha.W_K.bias.copy_(torch.zeros(d_model))
        mha.W_V.weight.copy_(torch.eye(d_model))
        mha.W_V.bias.copy_(torch.zeros(d_model))
        mha.W_O.weight.copy_(torch.eye(d_model))
        mha.W_O.bias.copy_(torch.zeros(d_model))
    x = torch.randn(B, S, d_model)
    assert torch.allclose(mha(x,x,x), attention(x,x,x))


    

    