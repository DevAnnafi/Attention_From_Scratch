# Day 5 — Why √d_k

**Goal:** Empirically verify the claim in footnote 4 of the paper, and measure
what it actually does to the softmax.

No new module today. This is measurement — confirming a derivation rather than
taking it on faith.

---

## The Prediction

From footnote 4 of *Attention Is All You Need*:

> Assume the components of q and k are independent random variables with mean 0
> and variance 1. Then their dot product has mean 0 and variance `d_k`.

Derived on Day 3:

- Variance of raw scores is `d_k`, so their **standard deviation is `√d_k`**.
- Dividing a random variable by `c` divides its variance by `c²`.
- So dividing by `√d_k` divides the variance by `d_k`, giving **variance 1**.

**Three testable claims:**

1. Raw `QK^T` variance ≈ `d_k`
2. Scaled variance ≈ 1, at every `d_k`
3. Unscaled softmax saturates as `d_k` grows; scaled softmax does not

---

## Setup

`torch.randn` produces exactly the distribution footnote 4 assumes — mean 0,
variance 1. Sequence length 100, so each run produces 10,000 scores, enough to
measure variance meaningfully.

```python
for d_k in [4, 64, 512]:
    Q = torch.randn(100, d_k)
    K = torch.randn(100, d_k)

    scores = Q @ K.transpose(0, 1)
    scaled = scores / math.sqrt(d_k)

    print(scores.var(), scaled.var())
    print(torch.softmax(scores, dim=-1).max(),
          torch.softmax(scaled, dim=-1).max())
```

---

## Results

| `d_k` | raw variance | scaled variance | unscaled max weight | scaled max weight |
|---:|---:|---:|---:|---:|
| 4 | 4.24 | 1.06 | 0.838 | 0.317 |
| 64 | 64.19 | 1.00 | 1.000 | 0.231 |
| 512 | 502.17 | 0.98 | 1.000 | 0.229 |

All three claims confirmed.

---

## What The Numbers Show

**Claim 1 — raw variance tracks `d_k`.** Across two orders of magnitude, the
variance of `QK^T` lands on `d_k` every time. Footnote 4 holds.

**Claim 2 — scaling normalizes it.** 1.06, 1.00, 0.98. Whatever `d_k` you pick,
the scores arriving at softmax have variance 1. This is what makes the model
scalable: `d_model` can change without the attention layer's behavior changing
with it.

**Claim 3 — the downstream effect is the whole point.** The unscaled max weight
climbs from 0.838 at `d_k = 4` to exactly 1.000 by `d_k = 64` — which is the
paper's own per-head dimension. Fully saturated at the size the architecture
actually uses.

Inspecting the unscaled softmax matrix at `d_k = 512` directly: values like
`1.7e-32`, and many entries that are literally `0.0000e+00` — underflowed to
zero. One key takes all the weight, the other 99 get nothing. That is not a
weighted average, it is a hard lookup, and gradients through those near-zero
weights are effectively nil.

The scaled column barely moves: 0.317, 0.231, 0.229. The strongest key gets
roughly a quarter of the attention and the rest spreads across the others — a
real weighted average with gradients that survive.

Same Q, same K, same softmax function. The only difference is one division.

---

## Two Things Worth Noting

**Saturation is a trend, not a switch.** At `d_k = 4` the unscaled max is 0.838
— lopsided but still functional. By 64 it is total. Running the small case is
what shows this; testing only at 512 would have made it look like a threshold
effect.

**Variance estimates concentrate as `d_k` grows.** Repeated runs at `d_k = 4`
gave 3.27, 3.71, 3.80, 4.10, 4.19, 4.46 — wandering by about 25%. At `d_k = 512`
the runs sat between 507 and 513, roughly 1%. More dimensions means more terms
in each dot product, so the sum concentrates around its expected value.

This is also why the experiment needed multiple runs at each `d_k` rather than
one draw. A single `d_k = 4` result of 3.27 could easily have looked like a
failed prediction.

---

## Connection Back

This is the mechanism behind the Day 2 warmup. `softmax([1, 2, 10])` collapsed
to `[0.0001, 0.0003, 0.9995]` because of the size of the gaps between inputs.
Growing `d_k` grows exactly those gaps — the raw scores spread out as `√d_k`,
so by `d_k = 64` every row looks like the `[1, 2, 10]` case.

The scaling factor pulls the inputs back to a fixed spread, so softmax sees the
same kind of distribution no matter how large the model gets.

---

## Open Question Carried Forward

Still have not derived what the gradient of softmax actually looks like when one
output is ~1 and the rest are ~0. The empirical case for saturation being bad is
now solid, but the gradient argument is still taken on the paper's word.