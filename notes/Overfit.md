# Day 14 — Overfit One Batch

**Goal:** Prove the full architecture works end to end by overfitting a single
batch to near-zero loss. No new modules — just a training loop.

If the model can memorize two examples, every module is wired correctly.

---

## The Task

Sequence copying — input `[a, b, c, d, e]`, target is the same sequence.
Trivial enough that a correct model should memorize it in a few hundred steps.
Hard enough that a broken model won't converge at all.

Two examples in the batch, sequence length 5, vocab size 20.

---

## The Result

```
step   0, loss 3.6075
step  10, loss 1.5601
step  20, loss 0.7880
step  50, loss 0.1957
step 100, loss 0.0726
step 200, loss 0.0248
step 300, loss 0.0127
step 400, loss 0.0078
step 490, loss 0.0055
```

3.6 → 0.005 in 500 steps. Consistent descent, near-zero at the end.

---

## What This Proves

Every module in the forward pass contributed correctly:

- `TokenEmbedding` — IDs → vectors, scaled by √d_model
- `PositionalEncoding` — sinusoid matrix added correctly
- `Encoder` — N layers of MHA + FFN + residual
- `Decoder` — N layers of masked self-attention + cross-attention + FFN + residual
- Cross-attention wiring — Q from decoder, K/V from encoder output
- `MultiHeadAttention` — projections, head splitting, attention, merging
- Final projection — `(B, S, d_model)` → `(B, S, vocab_size)`

A plateau or bouncing loss would have meant something upstream was broken.
Near-zero means the gradient path is intact through the whole network.

---

## The CrossEntropyLoss Shape Issue

`nn.CrossEntropyLoss` expects logits of shape `(N, C, d1)` — batch, classes,
sequence. The model outputs `(B, S, vocab_size)` — batch, sequence, classes.

Fix: transpose before passing to the loss.

```python
loss_input = torch.transpose(out, 1, 2)  # (B, vocab_size, S)
loss = criterion(loss_input, tgt)
```

The target stays as `(B, S)` — CrossEntropyLoss handles that shape directly.

---

## The Import Path Fix

Running scripts directly from `experiments/` fails with `ModuleNotFoundError`
because Python doesn't know about the repo root. Fix by inserting it at the
top of the script:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
```

This is different from pytest, which uses `conftest.py` for the same purpose.

---

## What Would Have Failed

If any of the following were wrong, loss would not have descended:

- Operand order in attention (`Q @ K^T` vs `K^T @ Q`) — Day 2's bug
- Wrong softmax dim — weights wouldn't normalize correctly
- Cross-attention K/V from wrong tensor — decoder couldn't look up source
- Dropped result in any forward chain — module's output silently discarded
- Missing `super().__init__()` — parameters not registered, optimizer updates nothing
- Missing `contiguous()` before `view` in merge_heads — runtime error

All of these were hit and fixed during the build. The clean loss curve is
confirmation they stayed fixed through assembly.

---

## Hyperparameters Used

```python
d_model   = 32
num_heads = 2
d_ff      = 64
N         = 3
vocab_size = 20
max_len   = 5
lr        = 1e-3  # Adam
steps     = 500
batch     = 2 examples, sequence length 5
```