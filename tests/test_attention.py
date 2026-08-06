import pytest
import torch
from src.attention import attention

def test_attention():
    Q = torch.tensor([[3.0,4.0], [8.0,2.0]])
    K = torch.tensor([[7.0,6.0], [5.0,9.0]])
    V = torch.tensor([[1.0,2.0], [3.0,4.0]])
    result = (attention(Q,K,V))
    expected = torch.tensor([[2.9717, 3.9717], [1.0017, 2.0017]])
    assert torch.allclose(result, expected, atol=1e-4)

    