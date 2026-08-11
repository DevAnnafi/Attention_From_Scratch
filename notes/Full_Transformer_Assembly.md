# Day 13 — Full Transformer Assembly

**Goal:** Assemble all components into a complete Transformer in `src/model.py`.
Two classes: `Decoder` stack and `Transformer`.

---

## What Gets Built

**`Decoder` stack** — wraps N `DecoderLayer` instances in an `nn.ModuleList`,
passing `encoder_output`, `src_mask`, and `tgt_mask` through each layer.

**`Transformer`** — the full model:

```
__init__:  d_model, num_heads, d_ff, N, vocab_size, max_len
           self.te      — TokenEmbedding (source)
           self.tgt_te  — TokenEmbedding (target)
           self.pe      — PositionalEncoding
           self.e       — Encoder stack
           self.d       — Decoder stack
           self.lp      — nn.Linear(d_model, vocab_size)

forward:   src, tgt, src_mask=None, tgt_mask=None
           es           = pe(te(src))
           encode       = e(es)
           embed_target = pe(tgt_te(tgt))
           decode       = d(embed_target, encode, src_mask, tgt_mask)
           return lp(decode)
```

Input: `src (B, S_src)`, `tgt (B, S_tgt)` — integer token IDs.
Output: `(B, S_tgt, vocab_size)` — logits over vocabulary.

---

## Two Separate Embeddings

Source and target each get their own `TokenEmbedding`. The paper shares weights
between the two embedding layers and the final projection (section 3.4), but
that is an optimization — two separate embeddings are correct and simpler to
build first.

One `PositionalEncoding` is shared between source and target since the formula
is the same for both.

---

## The Argument Order Bug

The `Decoder` stack's `ModuleList` initially contained `EncoderLayer` instances
instead of `DecoderLayer`. The encoder layer's `forward` only takes `x` — so
when the decoder loop passed four arguments, PyTorch raised:

```
TypeError: EncoderLayer.forward() takes 2 positional arguments but 5 were given
```

The traceback named `EncoderLayer` directly. Reading error messages top to
bottom and finding the first line that points at your own code is the right
debugging reflex — the PyTorch internals above it are noise.

**Fix:** change `EncoderLayer` to `DecoderLayer` in the `Decoder.__init__`
ModuleList.

---

## The Decoder Forward Argument Order

`self.d(embed_target, encode, src_mask, tgt_mask)` — not `self.d(encode, embed_target, ...)`.

The decoder stack's `forward` signature is `(x, encoder_output, src_mask, tgt_mask)`.
`x` is the target sequence being decoded; `encoder_output` is what cross-attention
looks up. Swapping them passes the encoder output through masked self-attention
and the target embeddings into cross-attention — plausible shapes, completely
wrong semantics.

This is the same class of error as the Day 2 operand order bug: same shape in,
wrong result, no error raised.

---

## Tests

`tests/test_model.py` — `test_transformer`:

```python
src = torch.randint(0, vocab_size, (B, S))
tgt = torch.randint(0, vocab_size, (B, S))
out = Transformer(d_model, num_heads, d_ff, N, vocab_size, max_len)(src, tgt)
assert out.shape == (B, S, vocab_size)
```

Inputs are integer tensors — token IDs, not floats. `torch.randint` not `torch.randn`.

---

## Status

The full architecture is assembled and passing a shape test. Day 14 is the real
proof of correctness: overfit a single batch, watch the loss go to near-zero,
and confirm the model can memorize before claiming it can generalize.