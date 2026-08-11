import torch
import torch.nn as nn
import math


def attention(q, k, v, mask=None, dropout=None):
    # Compute Q @ K^T
    compute = q @ torch.transpose(k, -1, -2)
    # Create d_k
    d_k = q.shape[-1]
    # Divide by radical d_k
    second_step = compute / math.sqrt(d_k)
    # Apply mask if provided
    if mask is not None:
        second_step = second_step.masked_fill(mask == 0, float('-inf'))
    # Softmax along the key axis
    softmax_result = torch.softmax(second_step, dim=-1)
    # Apply dropout if provided
    if dropout is not None:
        softmax_result = dropout(softmax_result)
    # Multiply by V
    mult_v = softmax_result @ v
    return mult_v


def split_heads(x, num_heads):
    B = x.shape[0]
    S = x.shape[1]
    d_model = x.shape[2]
    d_k = d_model // num_heads
    first_transform = x.reshape(B, S, num_heads, d_k)
    result = first_transform.transpose(1, 2)
    return result


def merge_heads(x):
    second_transform = x.transpose(1, 2).contiguous()
    B = x.shape[0]
    S = x.shape[2]
    H = x.shape[1]
    d_k = x.shape[3]
    last_transform = second_transform.view(B, S, H * d_k)
    return last_transform


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        q_proj = self.W_Q(q)
        k_proj = self.W_K(k)
        v_proj = self.W_V(v)
        split_q = split_heads(q_proj, self.num_heads)
        split_k = split_heads(k_proj, self.num_heads)
        split_v = split_heads(v_proj, self.num_heads)
        attn_out = attention(split_q, split_k, split_v, mask, self.dropout)
        merged = merge_heads(attn_out)
        o_proj = self.W_O(merged)
        return o_proj


if __name__ == "__main__":
    Q = torch.tensor([[3.0, 4.0], [8.0, 2.0]])
    K = torch.tensor([[7.0, 6.0], [5.0, 9.0]])
    V = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(attention(Q, K, V))
    x = torch.randn(2, 6, 8)
    print(split_heads(x, 2).shape)
    y = torch.randn(2, 6, 8)
    mha = MultiHeadAttention(8, 2)
    out = mha(y, y, y)
    print(out.shape)