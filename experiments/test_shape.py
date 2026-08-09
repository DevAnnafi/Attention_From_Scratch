import torch
from src.layers import FeedForward, LayerNorm

x = torch.randn(32, 100, 512)

ffn = FeedForward(d_model=512, d_ff=2048)
out = ffn(x)
print("FFN output shape:", out.shape)

ln = LayerNorm(d_model=512)
out2 = ln(x)
print("LayerNorm output shape:", out2.shape)