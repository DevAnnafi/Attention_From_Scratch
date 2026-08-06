# Day 2 — Attention Math by Hand

**Goal:** Compute scaled dot-product attention end to end by hand, then verify
against PyTorch. The result becomes the test case for `attention()` on Day 4.

---

## Softmax Warmup

Computed `softmax([1, 2, 3])` by hand:

```
[0.090, 0.245, 0.665]
```

Matches `torch.softmax` (0.0900, 0.2447, 0.6652).

**Observation:** the inputs are evenly spaced but the outputs are not — the gap
from 0.090 to 0.245 is smaller than the gap from 0.245 to 0.665. Softmax does
not just normalize, it amplifies differences.

Pushed further with `softmax([1, 2, 10])`:

```
[0.0001, 0.0003, 0.9995]
```

Effectively a hard argmax. Two consequences:

1. In attention, near-one-hot weights mean a token attends to exactly one other
   token and ignores the rest — destroying the ability to blend information
   across positions.
2. Gradients through a saturated softmax are tiny, so learning stalls.

This is the motivation for the `√d_k` scaling. Raw scores are dot products over
`d_k` terms, so their magnitude grows with `d_k`. Larger `d_k` → larger scores →
more saturated softmax. Day 5 measures this directly.

---

## The Max Subtraction Trick

Before exponentiating, subtract the row maximum from every entry in that row.

**Why it's valid:** softmax is invariant to adding a constant to all inputs —
multiplying numerator and denominator by `e^(-c)` cancels.

**Why it's necessary:** `e^48` is around 10^20. With realistic `d_k`, exponents
overflow to infinity and you end up dividing infinity by infinity. After the
subtraction the largest exponent is 0, so `e^0 = 1` and everything else falls
between 0 and 1.

PyTorch does this internally on every softmax call. My Day 4 implementation
needs it too, or it will produce NaNs once `d_k` is realistic.

Verified empirically: computed row 0 both with and without the subtraction and
got the same answer.

---

## The Worked Example

Sequence length 2, `d_k = 2`.

```
Q = [[3, 4],        K = [[7, 6],        V = [[1, 2],
     [8, 2]]             [5, 9]]             [3, 4]]
```

### Step 1 — Scores: `Q @ K^T`

```
[[45, 51],
 [68, 58]]
```

Entry `(i, j)` is row `i` of Q dotted with row `j` of K. Verified `(0,1)` by
hand: `[3,4] · [5,9] = 15 + 36 = 51`. ✓

Each entry is a dot product between a query vector and a key vector — the
similarity between them.

### Step 2 — Scale by `√d_k`

`√2 ≈ 1.414`

```
[[31.8198, 36.0624],
 [48.0833, 41.0122]]
```

Hand values agreed to the third decimal; PyTorch carries `√2` to full float32
precision, hence the small divergence. This is why correctness checks use
`torch.allclose` rather than `==`.

### Step 3 — Softmax along the key axis

Each row normalized independently, so each query's weights sum to 1.

Row 0: max is 36.0624 → `[-4.2426, 0]` → exponentiate → normalize

```
[0.014166, 0.98583]
```

Row 1: max is 48.0833 → `[0, -7.0711]` → exponentiate → normalize

```
[0.99915, 0.00084860]
```

Row 0's gap is ~4.2, row 1's is ~7.07 — and row 1 is correspondingly more
lopsided, consistent with the warmup.

Full weight matrix:

```
[[0.014166,  0.98583],
 [0.99915, 0.00084860]]
```

Both rows sum to 1. ✓

### Step 4 — Multiply by V

```
[[2.9717,  3.9717],
 [1.0017, 2.0017]]
```

PyTorch:

```
tensor([[2.9717, 3.9717],
        [1.0017, 2.0017]])
```

Exact match. ✓

**Sanity check on the values:** row 0 puts 98.6% of its weight on V's second
row, so the output's first row should be close to `[3, 4]` — it is. Row 1 puts
99.9% on V's first row, so it should be close to `[1, 2]` — it is.

---

## Day 4 Test Case

Assert `attention(Q, K, V)` produces:

```python
Q = [[3., 4.], [8., 2.]]
K = [[7., 6.], [5., 9.]]
V = [[1., 2.], [3., 4.]]

expected = [[2.9717, 3.9717],
            [1.0017, 2.0017]]
```

---

## What I Got Wrong

**1. Operand order in the matmul.** I computed `K^T @ Q` instead of `Q @ K^T` —
twice. The arithmetic was correct both times; the order was not. Matrix
multiplication is not commutative, so this produced a plausible-looking 2×2 with
no error raised. Same silent-failure pattern as Day 1's broadcasting bugs: the
shape is right, the numbers are wrong, and nothing complains.

**2. Transposing by swapping rows.** On the first attempt I swapped K's rows
rather than flipping across the diagonal. Related to the above — both are
"I know what the operation means but applied it wrong" rather than
misunderstanding the concept.

**3. Sign error in the max subtraction.** Wrote `[0, 7.0711]` instead of
`[0, -7.0711]`, putting the zero in the wrong slot. Caught because the resulting
weights favored the smaller score, which is backwards. Worth noting that the
*shape* of the error was detectable from the semantics, not the arithmetic.

**4. Wrong test case, marked as verified.** *(Found on Day 4.)* Recorded step 4
as `[[2.972, 3.972], [0.9944, 1.9872]]` with "Exact match ✓" underneath — row 1
was an arithmetic slip, row 0 was rounding error from carrying 4-place softmax
weights into the multiplication. The claimed PyTorch check could not have been
run on these numbers; it would have failed. Caught when the Day 4 implementation
disagreed and printing all four intermediates isolated the divergence to step 4.
A "✓" is a claim like any other, and writing one next to a check I did not run
made a bad test case look trustworthy for two days.

---

## The One That Will Matter Later

Operand order. `Q @ K^T` and `K^T @ Q` both return a valid matrix of the same
shape when the inputs are square, so nothing catches the mistake. In multi-head
attention the tensors are 4-D and the same class of error is far harder to spot
by eye. The defense is asserting on values against a known-good case — which is
exactly what the Day 4 test above is for.

---

## Open Questions

- What does the gradient of softmax actually look like when one output is 0.9995
  and the rest are near zero? Reasoned that it must be small, but haven't
  derived it.
- At what `d_k` does the saturation problem become severe in practice? Day 5.