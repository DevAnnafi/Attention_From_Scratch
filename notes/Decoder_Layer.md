# Day 12 — Decoder Layer

**Goal:** Implement `DecoderLayer` in `src/layers.py` — three sub-layers with
correct residual wiring and the right tensors flowing into cross-attention.

Hardest assembly day so far because cross-attention is asymmetric.

---

## Structure

```
__init__:  d_model, num_heads, d_ff
           self.ma   — MultiHeadAttention (masked self-attention)
           self.ca   — MultiHeadAttention (cross-attention)
           self.ffn  — FeedForward
           self.norm1, self.norm2, self.norm3

forward:   x, encoder_output, src_mask=None, tgt_mask=None
```

Three sub-layers in order:

```
x = norm1(x + masked_self_attention(x, x, x, tgt_mask))
x = norm2(x + cross_attention(x, encoder_output, encoder_output, src_mask))
x = norm3(x + ffn(x))
return x
```

---

## The Cross-Attention Wiring

The asymmetric one. From Day 3 notes:

| Sub-layer | Q from | K, V from | Mask |
|---|---|---|---|
| Masked self-attention | x | x | tgt_mask |
| Cross-attention | x | encoder_output | src_mask |
| FFN | — | — | — |

Q comes from the decoder (`x`) because that's what's being generated. K and V
come from `encoder_output` because the decoder is looking up information from
the encoded source.

Getting K and V wrong here is the single most common bug in a from-scratch
decoder. Both `MultiHeadAttention` calls take `(q, k, v, mask)` — the positions
matter.

---

## The Two Masks

**`tgt_mask`** — causal mask applied in masked self-attention. Prevents position
`t` from attending to positions after `t`. This is the Day 11 mask. Without it,
the decoder reads future tokens during training and collapses at inference.

**`src_mask`** — padding mask applied in cross-attention. Prevents the decoder
from attending to `<pad>` tokens in the encoder output. Different purpose,
different sub-layer.

Both default to `None` — the module works without them, which is useful for
shape testing.

---

## Naming

Started with both `Decoder` and `DecoderLayer` as identical classes in the file.
`Decoder` is reserved for the stack — same convention as `EncoderLayer` and
`Encoder`. The single layer is `DecoderLayer`; the stack wrapping N of them
comes on Day 13.

---

## What Goes in `__init__` vs Elsewhere

Tried putting `x = torch.randn(B, S, d_model)` inside `__init__`. The instinct
to catch: `__init__` builds modules the layer needs permanently. A random tensor
is test code — it belongs in the test function, not the class.

The question to ask before writing any `__init__` line: "is this a module the
layer needs permanently, or does it only exist during a forward pass or a test?"

---

## The Broken Chain Pattern

First `forward` attempt computed each sub-layer into a separate variable but
passed the original `x` into subsequent sub-layers instead of the updated one.
Cross-attention received the pre-norm-1 `x` instead of the post-norm-1 output,
and the FFN received the original `x` instead of the cross-attention output.

The fix: update `x` at each step rather than naming each intermediate. Makes
the chain impossible to break because there's only one variable to pass forward.

---

## Tests

`tests/test_layers.py` — `test_decoder_shape`:

Input: `x = torch.randn(B, S, d_model)` and `encoder_output = torch.randn(B, S, d_model)`.
Assert output shape `(B, S, d_model)` — unchanged.

Note: `d_ff` must be passed and should be sensible — conventionally larger than
`d_model`. A small arbitrary number like 7 doesn't cause a shape error but
doesn't reflect real usage.

---

## Status

Encoder stack and decoder layer both built. Day 13 assembles the full model:
decoder stack, final linear projection, and the complete encoder-decoder pipeline.
Day 14 is the overfit-one-batch proof of correctness.