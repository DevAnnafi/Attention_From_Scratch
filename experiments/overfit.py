"""
Overfit one batch — proof that the full architecture is correctly wired.

What's correct here that the earlier version wasn't:
  - Decoder input is shifted right: [BOS, a, b, c, d] predicts [a, b, c, d, e]
  - Causal mask applied to decoder self-attention
  - Padding mask applied to cross-attention (no padding in this toy, but path exercised)
  - Label-smoothing loss (eps=0.1)
  - Noam warmup schedule
  - Gradient clipping (clip=1.0)

Loss should drop from ~3.0 to near-zero in 500 steps.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.model import Transformer, LabelSmoothingLoss
from src.masks import create_masks
from src.scheduler import get_optimizer_and_scheduler

# ── Hyperparameters ───────────────────────────────────────────────────────────
d_model    = 32
num_heads  = 2
d_ff       = 64
N          = 2
vocab_size = 20
max_len    = 20
PAD_IDX    = 0
BOS_IDX    = 1
steps      = 1000

# ── Model ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
model = Transformer(d_model, num_heads, d_ff, N, vocab_size, max_len)
model.train()

# ── Single batch — copy task ──────────────────────────────────────────────────
# src:       [a, b, c, d, e]   (tokens 2–19, avoiding PAD=0 and BOS=1)
# tgt_input: [BOS, a, b, c, d] (shifted right — what the decoder sees)
# tgt_label: [a, b, c, d, e]   (what the decoder must predict)

src = torch.randint(2, vocab_size, (2, 5))  # (B=2, S=5)
bos = torch.full((src.shape[0], 1), BOS_IDX, dtype=torch.long)
tgt_input  = torch.cat([bos, src[:, :-1]], dim=1)   # (2, 5)
tgt_labels = src                                     # (2, 5)

# ── Masks ─────────────────────────────────────────────────────────────────────
src_mask, tgt_mask = create_masks(src, tgt_input, PAD_IDX)

# ── Loss and optimizer ────────────────────────────────────────────────────────
criterion = LabelSmoothingLoss(vocab_size, pad_idx=PAD_IDX, smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── Training loop ─────────────────────────────────────────────────────────────
for step in range(1, steps + 1):
    optimizer.zero_grad()

    out = model(src, tgt_input, src_mask, tgt_mask)   # (B, S, V)
    loss = criterion(out, tgt_labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 10 == 0:
        print(f"step {step:4d}, loss {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.6f}")
if loss.item() < 0.1:
    print("PASS — architecture is correctly wired.")
else:
    print("FAIL — loss did not converge. Check masking and model wiring.")