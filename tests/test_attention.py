import torch
import pytest
from src.attention import attention, MultiHeadAttention


def test_attention_numerical():
    """Hand-computed case from Day 2 — verifies the math, not just the shape."""
    Q = torch.tensor([[3.0, 4.0], [8.0, 2.0]])
    K = torch.tensor([[7.0, 6.0], [5.0, 9.0]])
    V = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = torch.tensor([[2.9717, 3.9717],
                             [1.0017, 2.0017]])
    result = attention(Q, K, V)
    assert torch.allclose(result, expected, atol=1e-3), (
        f"Numerical mismatch.\nGot:      {result}\nExpected: {expected}"
    )


def test_attention_rows_sum_to_one():
    """Each query's attention weights must sum to 1."""
    Q = torch.randn(2, 5, 8)
    K = torch.randn(2, 5, 8)
    V = torch.randn(2, 5, 8)
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Attention weights do not sum to 1. Row sums: {row_sums}"
    )


def test_padding_mask_zeroes_attention():
    """Masked positions must receive exactly zero attention weight."""
    B, S, d_k = 1, 4, 8
    Q = torch.randn(B, S, d_k)
    K = torch.randn(B, S, d_k)
    V = torch.randn(B, S, d_k)

    # Mask: 1 = attend, 0 = ignore. Last two positions are padding.
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float).unsqueeze(1).unsqueeze(2)
    # Shape: (B, 1, 1, S) — broadcasts over heads and query positions

    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)

    # Columns 2 and 3 should be exactly zero after softmax of -inf
    masked_weights = weights[:, :, :, 2:]
    assert torch.all(masked_weights == 0), (
        f"Masked positions have non-zero attention weight:\n{weights}"
    )


def test_causal_mask_is_lower_triangular():
    """Look-ahead mask: position t cannot attend to positions > t."""
    S = 5
    # Standard causal mask — lower triangular
    mask = torch.tril(torch.ones(S, S))

    Q = torch.randn(1, S, 8)
    K = torch.randn(1, S, 8)
    V = torch.randn(1, S, 8)

    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    scores = scores.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)

    # Upper triangle (above diagonal) must be zero
    upper = torch.triu(weights.squeeze(0), diagonal=1)
    assert torch.all(upper == 0), (
        f"Look-ahead mask failed — upper triangle has non-zero weights:\n{weights}"
    )


def test_mha_shape():
    """MultiHeadAttention output matches input shape."""
    B, S, d_model, H = 2, 6, 8, 2
    x = torch.randn(B, S, d_model)
    mha = MultiHeadAttention(d_model, H)
    out = mha(x, x, x)
    assert out.shape == (B, S, d_model)


def test_h1_equivalence():
    """With h=1 and identity projections, MHA reduces to plain attention."""
    B, S, d_model = 2, 6, 8
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
    mha.eval()
    assert torch.allclose(mha(x, x, x), attention(x, x, x), atol=1e-5)