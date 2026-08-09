# Day 10 — EncoderLayer

**Goal:** Implement `EncoderLayer` in `src/layers.py`, combining
Multi-Head Attention, FeedForward, and LayerNorm into one block. Verify
with pytest, not a scratch file, from the start.

Lives in `src/layers.py`, alongside `FeedForward` and `LayerNorm`.
Inherits from `nn.Module`.

---

## EncoderLayer

Reference: Section 3.1 of the paper — the encoder has two sublayers per
layer (self-attention, then FFN), each wrapped in a residual connection
and LayerNorm ("Add & Norm").

```
__init__:  d_model, num_heads, d_ff
           attention = MultiHeadAttention(d_model, num_heads)
           ffn = FeedForward(d_model, d_ff)
           norm1 = LayerNorm(d_model)
           norm2 = LayerNorm(d_model)

forward:   x1 = norm1(x + attention(x, x, x))
           x2 = norm2(x1 + ffn(x1))
           return x2
```

Input `(B, S, d_model)` → output `(B, S, d_model)`. Shape unchanged, same
as every sublayer inside it — this is what lets `EncoderLayer` stack
repeatedly to build the full encoder.

### Two sublayers, applied in sequence, not parallel

```
x  →  attention(x, x, x)  →  x1 = norm1(x + attention_out)
x1 →  ffn(x1)              →  x2 = norm2(x1 + ffn_out)
```

The second sublayer operates on `x1` — the *output* of the first block —
not the original `x`. Each sublayer builds on what the previous one
produced, so the FFN processes the combined features already shaped by
self-attention, rather than the raw input again.

### Add & Norm, unpacked

`LayerNorm(x + Sublayer(x))` is one nested expression: compute the
residual sum first, then pass the whole sum into `LayerNorm`. Written out
longhand for clarity:

```python
sublayer_out = self.attention(x, x, x)
combined = x + sublayer_out
x1 = self.norm1(combined)
```

Equivalent to `x1 = self.norm1(x + self.attention(x, x, x))` — same
computation, just unpacked into named steps.

### Self-attention: query, key, value all from x

`attention(x, x, x)` — the same tensor passed three times. Self-attention
means the sequence attends to itself, so query/key/value all come from
the same source. (Cross-attention, used in the decoder later, will pass
different tensors for each.)

### Why two separate LayerNorm instances

`norm1` and `norm2` can't share a single `LayerNorm` instance. `gamma`
and `beta` are learned parameters, and the attention output and the FFN
output have different distributions — each needs its own learned
scale/shift to adjust independently. One shared instance would force both
sublayers' outputs through the same normalization statistics, which
defeats the purpose of `gamma`/`beta` being learnable per-location.

### The divisibility constraint

`MultiHeadAttention` splits `d_model` into `num_heads` equal pieces, so
`d_model` must be divisible by `num_heads`. `d_ff` has no such
constraint — it's the FFN's internal expansion size and can be any value,
independent of the head split.

---

## Tests

`tests/test_layers.py` — written directly as pytest from the start this
time, rather than verified in a scratch file first (previous approach for
`FeedForward`/`LayerNorm` shape checks on Day 9).

```python
import torch
import torch.nn as nn
from src.layers import FeedForward, LayerNorm, EncoderLayer

def test_feed_forward():
    B, S, d_model, d_ff = 2, 6, 8, 5
    x = torch.randn(B, S, d_model)
    ff = FeedForward(d_model, d_ff)
    out = ff(x)
    assert out.shape == (B, S, d_model)

def test_layernorm_shape():
    B, S, d_model = 2, 6, 8
    x = torch.randn(B, S, d_model)
    ln = LayerNorm(d_model)
    out = ln(x)
    assert out.shape == (B, S, d_model)

def test_encoder_layer_shape():
    B, S, d_model, num_heads, d_ff = 2, 6, 8, 4, 5
    x = torch.randn(B, S, d_model)
    el = EncoderLayer(d_model, num_heads, d_ff)
    out = el(x)
    assert out.shape == (B, S, d_model)
```

All three inputs are `torch.randn` (floats), not `torch.randint`. The
only place `torch.randint` belongs across the whole codebase is
`TokenEmbedding`'s test — the one spot taking raw token IDs for a lookup
table. Every other class here — `PositionalEncoding`, `FeedForward`,
`LayerNorm`, `EncoderLayer` — operates on already-embedded, continuous
vectors.

`num_heads=4` chosen for `d_model=8` to satisfy the divisibility
constraint (8 ÷ 4 = 2 per head). `d_ff=5` picked freely — no constraint
applies to it.

All three pass:

```
tests/test_layers.py::test_feed_forward PASSED
tests/test_layers.py::test_layernorm_shape PASSED
tests/test_layers.py::test_encoder_layer_shape PASSED
3 passed in 1.10s
```

---

## What to Know for Later

`EncoderLayer` is one block. The full encoder (Section 3.1) stacks **N**
of these — the paper uses N=6 — with the output of one layer feeding
directly into the next. Not built yet.

The residual connection pattern (`x + Sublayer(x)`, wrapped in norm) is
the same shape everywhere it appears — encoder self-attention, decoder
self-attention, decoder cross-attention, and FFN blocks all use it. Once
this pattern is understood once, it explains every "Add & Norm" box in
the architecture diagram.

`self.d_model`, `self.num_heads`, `self.d_ff` are stored on
`EncoderLayer` but unused in `forward` — same pattern as `TokenEmbedding`
(Day 8) and `FeedForward`/`LayerNorm` (Day 9). Consistent, not a bug,
worth a single cleanup pass later if the file grows.