import torch
import math

def attention(q,k,v, mask=None):
    # Compute Q @ K^T
    compute = q @ torch.transpose(k, 0, 1)
    # Create d_k
    d_k = q.shape[-1]
    # Divide by radical d_k
    second_step = compute / math.sqrt(d_k)
    # Softmax along the key axis
    softmax_result = torch.softmax(second_step, dim=-1)
    # Multiply by V
    mult_v = softmax_result @ v

    return mult_v

Q = torch.tensor([[3.0,4.0], [8.0,2.0]])
K = torch.tensor([[7.0,6.0], [5.0,9.0]])
V = torch.tensor([[1.0,2.0], [3.0,4.0]])

print(attention(Q, K, V))

