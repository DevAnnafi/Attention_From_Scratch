# Day 8 — Embeddings and Positional Encoding

**Goal:** Implement `TokenEmbedding` and `PositionalEncoding` in `src/embedding.py`.
Verify the PE matrix visually before writing tests.

Both live in `src/embedding.py`. Both inherit from `nn.Module`.

---

## TokenEmbedding

Wraps `nn.Embedding` with the `√d_model` scaling from section 3.4.

```
__init__:  vocab_size, d_model
           nn.Embedding(vocab_size, d_model)
           scale = √d_model

forward:   return embedding(x) * scale
```

Input `(B, S)` of integer token IDs → output `(B, S, d_model)`.

**Why the scale:** embedding weights initialize small — values roughly ±1.
Positional encodings are sines and cosines, also in the range [-1, 1]. Added
together unscaled, positional signal would be comparable in magnitude to token
identity. Scaling the embeddings up makes token content dominate and lets
position act as a modifier. The paper states the factor without justifying it;
the reasoning above is the generally accepted explanation.

---

## PositionalEncoding

Computes the sinusoid matrix once in `__init__` and registers it as a buffer.
`forward` adds the relevant slice to the input.

```
__init__:  d_model, max_len
           pe = zeros(max_len, d_model)
           position = arange(0, max_len)
           div_term = exp(arange(0, d_model, 2) * -(log(10000) / d_model))
           pe[:, 0::2] = sin(position.unsqueeze(1) * div_term)
           pe[:, 1::2] = cos(position.unsqueeze(1) * div_term)
           register_buffer('pe', pe)

forward:   S = x.shape[1]
           return x + pe[:S]
```

Input `(B, S, d_model)` → output `(B, S, d_model)`. Shape unchanged — PE is
additive.

### Why register_buffer

The PE matrix is computed once and never updated — no gradient, no learning.
`register_buffer` stores it permanently on the module so it moves to GPU with
the model and gets saved in checkpoints, but does not appear in
`model.parameters()` and does not receive gradients.

Storing it as a plain tensor attribute would lose it on GPU transfer.
Storing it as a parameter would make the optimizer try to learn it, which is
wrong — the encoding is fixed by the formula.

### The div_term formula

`torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))`

Equivalent to the paper's `1 / 10000^(2i/d_model)` but computed in log space
to avoid overflow. The arange steps by 2 to produce only the even indices —
`[0, 2, 4, ..., d_model-2]` — since even and odd columns are handled separately.

### The unsqueeze

`position` is shape `(max_len,)` and `div_term` is shape `(d_model/2,)`. Multiplied
directly, broadcasting would align from the right and likely error if the sizes
differ.

`position.unsqueeze(1)` reshapes to `(max_len, 1)`. Broadcasting against
`(d_model/2,)` then produces `(max_len, d_model/2)` — every position paired with
every dimension index. That outer product is what the formula requires.

### Why small dimensions oscillate faster

Small `i` means `2i/d_model` is small, so `10000^(2i/d_model)` is close to 1,
so the denominator is small, so the argument to sine/cosine is large — meaning
the values change quickly with position. Large `i` produces a large denominator
and slow variation.

The heatmap makes this obvious: the div_term values decrease from left to right - 1.0 to 0.0001 for d_model = 8. A larger div_term means a larger argument to sine, so the left columns oscillate fast and the right columns barely change.

---

## Visual Verification

Plotted the PE matrix as a heatmap with `d_model=64, max_len=100`.

Correct output: banded pattern with high-frequency oscillation on the left
slowing toward the right, values ranging fully between -1 and 1.

This is the verification the formula is right. Uniform color or noise means
the formula is wrong. Getting here before writing the tests means a passing
test confirms the shape, not just that the code ran.

---

## Tests

`tests/test_embedding.py`:

**`test_token_embedding_shape`** — input `torch.randint(0, vocab_size, (B, S))`,
assert output shape `(B, S, d_model)`. Input must be integers in range —
`torch.randint` rather than `torch.randn`.

**`test_positional_encoding_shape`** — input `torch.randn(B, S, d_model)`,
assert output shape unchanged.

Both deliberately broken and confirmed red before being restored.

---

## What to Know for Later

`self.vocab_size` and `self.d_model` were stored unnecessarily in
`TokenEmbedding.__init__` — `forward` doesn't need them. Not a bug, just
extra lines. Worth cleaning if the file grows.

The PE matrix slice `self.pe[:S]` is shape `(S, d_model)`. Adding it to
`x` of shape `(B, S, d_model)` works because broadcasting expands the missing
batch dimension. This is the Day 1 "does not exist" clause firing correctly.