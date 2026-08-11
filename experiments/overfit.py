import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.model import Transformer

d_model = 32
num_heads = 2
d_ff = 64
N = 3
vocab_size = 20
max_len = 5

model = Transformer(d_model, num_heads, d_ff, N, vocab_size, max_len)

src = torch.randint(0, vocab_size, (2,5))
tgt = torch.randint(0, vocab_size, (2,5))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(500):
    optimizer.zero_grad()
    out = model(src, tgt)
    loss_input = torch.transpose(out, 1, 2)
    loss = criterion(loss_input, tgt)
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step}, loss {loss.item():.4f}")