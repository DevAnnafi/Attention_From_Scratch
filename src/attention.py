import torch
import torch.nn as nn
import math

# Large finite negative used instead of -inf.
# With padded batches, a pad query row attends only to pad keys.
# softmax([-inf, -inf, ...]) = NaN, which poisons every gradient.
# A large finite negative (-1e9) gives softmax ≈ 0 without NaN.
_MASK_VALUE = -1e9


def attention(q, k, v, mask=None, dropout=None):
    """Scaled dot-product attention.

    Args:
        q:       (..., S_q, d_k)
        k:       (..., S_k, d_k)
        v:       (..., S_k, d_v)
        mask:    broadcastable to (..., S_q, S_k); 0 = mask, 1 = keep
        dropout: nn.Dropout or None

    Returns:
        (..., S_q, d_v)
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., S_q, S_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, _MASK_VALUE)

    weights = torch.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    return weights @ v


def split_heads(x, num_heads):
    """(B, S, d_model) -> (B, H, S, d_k)"""
    B, S, d_model = x.shape
    d_k = d_model // num_heads
    return x.reshape(B, S, num_heads, d_k).transpose(1, 2)


def merge_heads(x):
    """(B, H, S, d_k) -> (B, S, d_model)"""
    B, H, S, d_k = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, H * d_k)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        q_proj = split_heads(self.W_Q(q), self.num_heads)
        k_proj = split_heads(self.W_K(k), self.num_heads)
        v_proj = split_heads(self.W_V(v), self.num_heads)
        attn_out = attention(q_proj, k_proj, v_proj, mask, self.dropout)
        return self.W_O(merge_heads(attn_out))