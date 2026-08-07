# Day 6 — Splitting and Merging Heads

**Goal:** Get `(B, S, d_model)` into `(B, H, S, d_k)` and back again, with the
round-trip verified. No attention, no projections — just the reshape gymnastics.

This is Day 1 §1.5 with real names.

---

## The Two Operations

**Split:** `(B, S, d_model)` → `(B, H, S, d_k)`

Two steps, and they cannot be collapsed into one:

1. `reshape(B, S, H, d_k)` — split the last dimension in two, leave S alone
2. `.transpose(1, 2)` — swap H and S

The single-reshape shortcut `(B, S, d_model) → (B, H, S, d_k)` looks like it
should work and does not. Reshape fills elements in the order they currently sit
in memory, so jumping straight to the target gives the right *shape* with
scrambled *contents* — no error raised. Exactly the bug that made the Day 1
round-trip return False.

**Merge:** `(B, H, S, d_k)` → `(B, S, d_model)`

Reverse each operation, in reverse order:

1. `.transpose(1, 2)` — undo the swap
2. `reshape(B, S, H * d_k)` — merge the last two dimensions

`d_model` is recovered as `H * d_k`, so `merge_heads` needs no `num_heads`
argument — the 4-D shape already carries the information. `split_heads` does
need it, because a `(B, S, d_model)` tensor cannot know how you want it divided.

---

## Why H Ends Up at Position 1

The head dimension moves in front of the sequence dimension so that each head
holds a complete `(S, d_k)` matrix. Attention then runs on the last two
dimensions, with B and H as batch dimensions that broadcast over — which is
exactly what batched matmul does with >2D inputs (Day 1 §3).

---

## The Contiguity Finding

`view` works on the final merge, which is *not* what Day 1 would predict — there,
a transposed tensor was non-contiguous and `view` failed on it.

Checked directly with `is_contiguous()` rather than reasoning about it: **True**.

**Why:** `split_heads` transposes 1↔2, and `merge_heads` transposes 1↔2 again.
Nothing happens in between, so the second transpose exactly undoes the first and
restores standard layout. Same finding as Day 1 §1.5 — transpose swaps strides,
it does not set a contiguity flag.

**Warning for Day 7:** this holds *only* because nothing runs between the two
transposes. On Day 7 attention runs in the middle, and whatever it returns may
have a different layout. The Day 1 `k = z * 1` experiment showed that an
elementwise op propagates the input's stride pattern rather than normalizing it.

So the stride error may appear on Day 7. If it does, it is not a bug in the merge
— it means the assumption about layout no longer holds, and `.contiguous()`
before the `view` is the fix.

Keeping `view` rather than `reshape` deliberately: `view` fails loudly when the
assumption breaks, `reshape` silently copies and moves on. Day 1 §1.4 made that
argument; this is the first place it actually applies.

---

## Tests

Two separate tests, in `tests/test_heads.py`:

- `test_split_heads_shape` — asserts `(B, H, S, d_model // H)`
- `test_head_round_trip` — asserts `torch.allclose(x, merge_heads(split_heads(x, H)))`

**Why two and not one:** shape and value-ordering are independent failure modes.
The Day 1 round-trip returned False with a perfectly correct shape. One combined
test would only say "something is wrong."

Both were deliberately broken and confirmed to fail before being restored.

Note: breaking the round-trip test by passing a wrong argument count produced a
`TypeError` — the test crashed *before* reaching the assert, which proves
nothing about the assertion. Had to break the comparison itself instead
(comparing `x` against `split`) to see a real failure.

---

## What I Got Wrong

**1. Dropping results.** Wrote `x.reshape(...)` without assigning it, then
transposed `x` instead of the reshaped tensor. Four separate times today. PyTorch
ops return new tensors; they do not modify in place. If the result is not caught,
it is gone.

**2. Hardcoding dimensions.** First version had `B = 2`, `S = 6`, `d_model = 8`
written as literals inside `split_heads`, and a hardcoded `reshape(2,2,6,4)`.
That makes the function work on exactly one tensor shape. Dimensions have to be
read off the input.

**3. `.shape` called as a function.** Wrote `x.shape()` and `x.shape(0,1,2)`.
It is an attribute holding a tuple, indexed with square brackets — never called.

**4. `x.shape[2 * 3]`.** Tried to multiply *indices* rather than the values at
those positions. Needed two separate lookups multiplied together.

**5. Swapped S and H repeatedly.** In `merge_heads` the input is `(B, H, S, d_k)`
— H at position 1, S at position 2. Got these backwards several times, in both
directions.

**6. Calling functions as methods.** Wrote `x.split_heads(H)` in the test.
These are plain functions: `split_heads(x, H)`.

---

## The One That Will Matter Later

The dropped-result pattern. It produces no error — the previous variable is still
valid, so the next line runs happily on the wrong tensor. On Day 7 the chain gets
longer (project → split → attend → merge → project) and a dropped step in the
middle will be much harder to spot than it was here.