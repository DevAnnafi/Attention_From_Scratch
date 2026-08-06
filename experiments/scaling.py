import torch
import math

for d_k in [4,64,512]:
    Q = torch.randn(100, d_k)
    K = torch.randn(100, d_k)

    result = Q @ torch.transpose(K, 0, 1)

    print(result.var())

    divide = result / math.sqrt(d_k)

    print(divide.var())

    softmax_unscaled = torch.softmax(result, dim=-1)
    softmax_scaled = torch.softmax(divide, dim=-1)

    print(softmax_unscaled.max())
    print(softmax_scaled.max())