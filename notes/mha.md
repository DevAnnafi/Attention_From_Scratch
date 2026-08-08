# Day 7 — Multi-Head Attention

**Goal:** Wrap Day 6's split/merge and Day 4's attention into a single
`nn.Module` with the four learned projections.

First `nn.Module` in the project.

---

## The Module

```
__init__:  d_model, num_heads
           W_Q, W_K, W_V, W_O  — all nn.Linear(d_model, d_model)

forward:   project q, k, v
           split each into heads
           attention
           merge heads
           project through W_O
```

Input `(B, S, d_model)`, output `(B, S, d_model)`. Same shape in, same shape out
— which is what lets encoder layers stack.

---

## Two Things About the Constructor

**`super().__init__()` comes first.** `nn.Module` does bookkeeping in its own
constructor — registries for parameters and child modules, training-mode flags.
Skipping it means assigned layers never get registered, so `model.parameters()`
comes back empty and the optimizer has nothing to update. The model would train
with the loss never moving.

**Why the projections are `d_model → d_model`, not `d_model → d_k`.** The paper's
formula shows per-head projections `W_Q_i` of shape `d_model × d_k` — eight
separate small matrices for `h = 8`.

But eight `512 × 64` matrices stacked side by side *are* one `512 × 512` matrix.
So a single wide Linear followed by `split_heads` produces the same result as
eight small ones, in one matmul instead of eight. That is the mechanism behind
the paper's claim that multi-head costs about the same as single-head at full
dimensionality (Day 3 notes, §3.2.2).

---

## Bug 1 — Hardcoded Transpose Dimensions

`attention()` was written and tested on 2-D tensors and contained:

```python
compute = q @ torch.transpose(k, 0, 1)
```

Correct for 2-D. On the 4-D `(B, H, S, d_k)` tensors coming out of `split_heads`,
it swaps B and H — completely wrong axes.

```
RuntimeError: Expected size for first two dimensions of batch2 tensor
to be: [4, 4] but got: [4, 6].
```

**Fix:** `torch.transpose(k, -2, -1)`.

**Why negative indices are the right call here:** on a 2-D tensor, -2 and -1 are
positions 0 and 1 — identical to the original behavior. On 4-D they are positions
2 and 3, leaving B and H alone as batch dimensions that broadcast. One expression
covers both ranks.

Same reasoning as `dim=-1` in the softmax: name the axis relative to the end, and
the code stops caring how many leading dimensions there are.

Confirmed the Day 4 test still passes after the change — the 2-D case is
unaffected.

---

## Bug 2 — The Stride Error, Predicted 24 Hours Early

```
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces).
Use .reshape(...) instead.
```

Raised at the `view` inside `merge_heads`.

**Why it appeared now and not on Day 6.** Yesterday the two transposes were
adjacent — `split_heads` transposed 1↔2 and `merge_heads` transposed straight
back, restoring standard layout, so `view` was legal. Today `attention` runs in
between, and what it returns does not have the layout the second transpose
assumed.

The Day 6 notes called this exactly:

> **Warning for Day 7:** this holds *only* because nothing runs between the two
> transposes. [...] So the stride error may appear on Day 7. If it does, it is
> not a bug in the merge — it means the assumption about layout no longer holds,
> and `.contiguous()` before the `view` is the fix.

**Fix:** `.contiguous()` before the `view`.

**Why not switch to `reshape`:** the Day 6 notes made the argument already —
`view` fails loudly when the contiguity assumption breaks, `reshape` silently
copies and moves on. The loud failure is what surfaced this in ten seconds
instead of never. Keeping `view`.

This is the Day 1 §1.4 lesson paying off for the second time.

---

## Tests

Three now cover the module:

**`test_multiheadattention`** — output shape equals input shape.

**`test_h1_equivalence`** — the one that actually catches scrambling.

Construct `MultiHeadAttention(d_model, 1)` and overwrite all four Linear layers
to be pass-throughs: identity weight, zero bias, applied in place with `copy_()`
inside `torch.no_grad()`. `nn.Linear` computes `xW^T + b`, so identity weight and
zero bias make the layer return its input unchanged.

With one head and no projections, the module should be doing *nothing* except
calling `attention`. So:

```python
assert torch.allclose(mha(x, x, x), attention(x, x, x))
```

**Why this test matters more than the shape test:** if split or merge reorders
elements, every shape stays correct and the values are wrong. That is precisely
what happened in the Day 1 §1.5 round-trip. A shape assertion cannot see it.

Broke it by changing the head count to 2 — the module no longer reduces to
single-head attention, values diverge, test goes red. Confirms the test is
sensitive to the thing it is meant to check, not just passing by accident.

---

## The Technique Worth Reusing

Reduce a complex module to a simpler one already verified, force them into the
same configuration, and assert they agree.

Cheap to set up, and it catches an entire class of bug — ordering, indexing,
axis mistakes — that shape assertions are blind to. Worth doing for the encoder
layer and the full model later.

---

## Smaller Things

**Dropped-result pattern, again.** First draft of `forward` merged `split_q`,
`split_k`, and `split_v` instead of the attention output — three lines that
undid the splits and discarded the attention result entirely. No error; the
variables all existed. Day 6 notes flagged this as the one that would matter,
and the longer chain here is exactly where it got harder to see.

**Naming collision.** Wrote `attention = attention(...)` inside `forward`,
shadowing the function being called. Renamed to `attn_out`.

**Module-level test code.** Loose `print` statements and tensor definitions at
module scope execute on every import, including from the test suite. Moved them
under `if __name__ == "__main__":`. Second time this has come up — the leftover
`print` in Day 6's commit was the first.

---

## Status

Hardest module in the architecture is built and verified.

Remaining: positional encoding (Day 8), FFN and LayerNorm (Day 9), encoder
assembly (Day 10), masking (Day 11), decoder (Day 12), full model (Day 13),
overfit-one-batch (Day 14).