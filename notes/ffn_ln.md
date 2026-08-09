# Day 9 — FeedForward and LayerNorm

**Goal:** Implement `FeedForward` and `LayerNorm` in `src/layers.py`.
Verify shapes with a quick script before committing.

Both live in `src/layers.py`. Both inherit from `nn.Module`.

---

## FeedForward

Reference: "Attention Is All You Need," Section 3.3.

```
__init__:  d_model, d_ff
           linear1 = nn.Linear(d_model, d_ff)
           linear2 = nn.Linear(d_ff, d_model)
           relu = nn.ReLU()

forward:   return linear2(relu(linear1(x)))
```

Input `(B, S, d_model)` → output `(B, S, d_model)`. Shape unchanged —
`d_ff` is only used internally.

```
Input:  (B, S, d_model)
           ↓ linear1: d_model → d_ff
        (B, S, d_ff)
           ↓ ReLU
        (B, S, d_ff)
           ↓ linear2: d_ff → d_model
Output: (B, S, d_model)
```

**Why d_model and d_ff are parameters, not hardcoded:** they describe the
width of the data at each stage, not the layers themselves. Passing them
in rather than writing `nn.Linear(512, 2048)` means the same class works
for any model size — e.g. `d_model=512, d_ff=2048` vs.
`d_model=768, d_ff=3072` — without touching the code.

**Why the FFN preserves shape:** the representation is temporarily
expanded to `d_ff`, passed through a nonlinearity, then projected back
down to `d_model`. This matters because FFN blocks need to stack with
attention blocks and residual connections without shape mismatches.

**Note:** `nn.Linear` layers are fully connected / affine transformations,
not convolutions — easy to misname in an interview.

---

## LayerNorm

Reference: Ba et al., 2016, "Layer Normalization" (the Transformer paper
only cites "Add & Norm," it doesn't derive LayerNorm itself).

```
__init__:  d_model
           ln = nn.LayerNorm(d_model)

forward:   return ln(x)
```

Input `(B, S, d_model)` → output `(B, S, d_model)`. Shape unchanged.

### The formula

```
output = gamma * (x - mean) / sqrt(variance + eps) + beta
```

Worked by hand on `x = [2.0, 4.0, 6.0, 8.0]`:

- mean = 5
- deviations: `[-3, -1, 1, 3]` → squared: `[9, 1, 1, 9]` → variance = 5
- `sqrt(variance) ≈ 2.236`
- `x_normalized = [-1.34, -0.45, 0.45, 1.34]` — mean 0, variance 1

### Why gamma and beta

Forcing every vector to mean=0, variance=1 is rigid and can destroy
information the network needs. `gamma` (rescale) and `beta` (reshift) are
learnable parameters, same shape as `d_model`, that let the network undo
or adjust that stabilization if a different scale or mean is more useful
at that layer. They initialize to `gamma=1, beta=0` — a no-op — and are
learned via backprop like any other weight. `nn.LayerNorm(d_model)`
creates both automatically at that size.

### Why per-token, not per-batch

LayerNorm computes a separate mean/variance for each token vector,
independent of every other token and every other example in the batch.
For input `(32, 100, 512)` that's 3200 independent (mean, variance)
pairs, each computed over one token's 512 values.

This matters because Transformer sequences have different lengths and
get padded to batch together. If statistics were computed across the
batch, padding tokens could pollute real tokens' statistics, and a
sentence's processing would depend on what else happened to be in its
batch. Per-token normalization avoids this — each token is self-contained
regardless of batch size, sequence length, or padding.

---

## Where this fits — Add & Norm (preview, not built yet)

Section 3.1 pattern, applied around every sublayer (attention or FFN):

```
output = LayerNorm(x + Sublayer(x))
```

`x +` is a **residual connection**: the original input is added back to
the sublayer's output before normalizing. Not implemented yet — comes
when the full encoder layer is assembled.

---

## Shape Verification

```python
x = torch.randn(32, 100, 512)

ffn = FeedForward(d_model=512, d_ff=2048)
out = ffn(x)           # torch.Size([32, 100, 512])

ln = LayerNorm(d_model=512)
out2 = ln(x)            # torch.Size([32, 100, 512])
```

Both preserve input shape, confirmed by running the script. Verification
done before committing, same as Day 8 — a passing shape check confirms
the code ran, but the derivation above is what confirms it's *correct*.

---

## What to Know for Later

`self.d_model` (and `self.d_ff` in `FeedForward`) are stored but never
used in `forward` — same pattern as `self.vocab_size`/`self.d_model` in
`TokenEmbedding` on Day 8. Not a bug, just extra lines worth cleaning up
if the file grows.

In practice, `LayerNorm` as its own wrapper class is usually skipped —
most implementations just call `nn.LayerNorm(d_model)` directly inside a
larger block, since there's no extra logic to justify a separate class.
Building it standalone today was for seeing construction and forward
explicitly, same reasoning as wrapping `TokenEmbedding` on Day 8.

`FeedForward` and `LayerNorm` are both shape-preserving. That's not a
coincidence — every sublayer in the encoder/decoder needs to preserve
`(B, S, d_model)` so blocks can stack via residual connections.