# Day 11 — Encoder

**Goal:** Implement `Encoder` in `src/layers.py` — a stack of N
`EncoderLayer`s, chained so each layer's output feeds the next. Verified
with pytest from the start, same as Day 10.

Lives in `src/layers.py`, alongside `FeedForward`, `LayerNorm`, and
`EncoderLayer`. Inherits from `nn.Module`.

---

## Encoder

Reference: Section 3.1 — the encoder is a stack of N=6 identical layers.
`Encoder` doesn't introduce new math; it wires together N copies of
`EncoderLayer` (already built, Day 10) in sequence.

```
__init__:  d_model, num_heads, d_ff, N
           layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff)
                                    for _ in range(N)])

forward:   result = x
           for layer in layers:
               result = layer(result)
           return result
```

Input `(B, S, d_model)` → output `(B, S, d_model)`. Shape unchanged —
same as every piece built so far, which is exactly what makes stacking
possible: each `EncoderLayer`'s output is a valid input to the next.

### Why nn.ModuleList, not a plain Python list

```python
self.layers = [EncoderLayer(...) for _ in range(N)]     # WRONG
self.layers = nn.ModuleList([EncoderLayer(...) for _ in range(N)])  # RIGHT
```

`nn.Module` has machinery that recognizes submodules assigned directly as
attributes (like `self.linear1 = nn.Linear(...)`) and registers them —
so their weights appear in `.parameters()`, move with `.to(device)`, and
save in `state_dict()`. A plain Python list is invisible to that
machinery: none of the layers inside it would be tracked, so the
optimizer would never update their weights and `.to(device)` wouldn't
move them. `nn.ModuleList` is a list-like container built specifically to
be recognized correctly — indexable and iterable like a list, but every
module inside it participates in the standard `nn.Module` bookkeeping.

### Chaining N layers — same pattern as chaining 2 sublayers

`EncoderLayer.forward` (Day 10) fed `x1` — the output of the first
sublayer — into the second sublayer, not the original `x`. `Encoder`
applies the identical idea across N layers instead of 2:

```python
result = x
for layer in self.layers:
    result = layer(result)   # each layer builds on the previous output
return result
```

`result` must be initialized to `x` *before* the loop starts — the loop
body only reassigns it, it doesn't create it. Missing that line causes a
`NameError` on the first iteration.

### N vs. the d_model/num_heads constraint

`N` is unrelated to the `d_model`/`num_heads` divisibility constraint.
`N` is just a layer count — any positive integer works, no math
relationship to `d_model`. The constraint from `MultiHeadAttention`
(`d_model` divisible by `num_heads`) still applies, but only because
`EncoderLayer` contains attention internally — it has nothing to do with
how many `EncoderLayer`s get stacked.

---

## Tests

Added to `tests/test_layers.py`, same file and pattern as Day 10:

```python
def test_encoder_shape():
    B, S, d_model, num_heads, d_ff, N = 2, 6, 8, 4, 7, 3
    x = torch.randn(B, S, d_model)
    e = Encoder(d_model, num_heads, d_ff, N)
    out = e(x)
    assert out.shape == (B, S, d_model)
```

`N=3` used for the test — small and fast, no need to match the paper's
N=6 just to verify shape correctness. `d_model=8, num_heads=4` keeps the
divisibility constraint satisfied.

All four tests in the file pass together:

```
test_feed_forward PASSED
test_layernorm_shape PASSED
test_encoder_layer_shape PASSED
test_encoder_shape PASSED
```

---

## What to Know for Later

The encoder side of the Transformer is now structurally complete:
embeddings (Day 8) → N-stacked encoder layers (Days 9–11). What's left on
the encoder side is largely assembly and configuration, not new concepts.

The decoder will reuse almost everything built here — `FeedForward`,
`LayerNorm`, the residual/Add & Norm pattern, and an `nn.ModuleList` stack
— but adds a second attention sublayer per layer (masked self-attention,
then cross-attention over the encoder's output) and a causal mask so a
position can't attend to future tokens. Not built yet.

`self.d_model`, `self.num_heads`, `self.d_ff`, `self.N` are stored on
`Encoder` but unused in `forward` — same recurring pattern as every class
this week (Days 8–10). Still just extra lines, not a bug.