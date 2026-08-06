# Day 3 — Reading the Paper

**Source:** Vaswani et al., *Attention Is All You Need* (arXiv 1706.03762)

**Goal:** Understand the architecture well enough to draw it from memory and
fill in the shape table.

**Status:** Section 3 complete (3.1–3.5). Diagram and shape table still to do.

---

## What to read (and what to skip)

For implementation purposes:

- **3.1–3.3** — architecture, attention, FFN. Core.
- **3.4** — embeddings and the final softmax layer.
- **3.5** — positional encoding. Needed for Day 8.
- **5.4** — dropout placement and label smoothing. Only matters once training.
- **5.3** — learning rate schedule with warmup. Post-Day-14.

Skippable for building: section 2 (related work), section 4 (why self-attention),
most of 6 (results), section 7.

Roughly five pages of load-bearing text.

---

## 3.1 — Encoder and Decoder Stacks

Encoder layer has two sub-layers. Decoder layer has three — the extra one is
cross-attention.

---

## 3.2.1 — Scaled Dot-Product Attention

This is the computation done by hand on Day 2, so the notation maps directly
onto a worked example.

### Why divide by √d_k

Main text: for large `d_k` the dot products grow large in magnitude, pushing
softmax into regions with extremely small gradients.

This matches what the Day 2 experiment showed — `softmax([1, 2, 10])` collapsed
to `[0.0001, 0.0003, 0.9995]`, effectively a hard argmax with near-zero
gradients everywhere.

Footnote 4 gives the statistical justification:

> Assume the components of q and k are independent random variables with mean 0
> and variance 1. Then their dot product has mean 0 and variance `d_k`.

Closing the loop:

- Variance is `d_k`, so **standard deviation is `√d_k`**.
- Dividing a random variable by `c` divides its variance by `c²`.
- So dividing by `√d_k` divides the variance by `d_k`, giving variance 1.

That is why the factor is `√d_k` and not `d_k` or anything else — it normalizes
the scores to unit variance regardless of model dimension. Whatever `d_k` you
pick, softmax sees inputs at a consistent scale, so it never saturates and
gradients survive.

Day 5 measures this empirically.

---

## 3.2.2 — Multi-Head Attention

**Why multiple heads:** multi-head attention lets the model jointly attend to
information from different representation subspaces at different positions. With
a single head, averaging inhibits this — distinct relationships get blurred
together. Partitioning preserves them; one head might track syntactic
dependencies while another tracks positional proximity.

**Dimensions:** `d_k = d_v = d_model / h`

With the base model's `h = 8` and `d_model = 512`, each head gets 64 dimensions.

**Compute cost:** similar to single-head attention at full dimensionality.
8 heads × 64 dims = 512, the same total width as one full-width head. Multi-head
is not adding capacity, it is partitioning it.

**Note:** `d_model = 512` is not stated in 3.2.2 — it has to be back-derived from
`d_model/h = 64` with `h = 8`, or found in 3.3 / Table 3 where it appears
explicitly.

**Constraint for implementation:** `h` must divide `d_model` evenly.

**Day 6 relevance:** splitting heads means reshaping `(B, S, 512)` →
`(B, S, 8, 64)` → transpose → `(B, 8, S, 64)`. This is exactly the round-trip
exercise from Day 1 §1.5, with real numbers.

---

## 3.2.3 — The Three Uses of Attention

The highest-value part of the section. Getting these wrong is the most common
bug in a from-scratch decoder.

| Where | Q from | K, V from |
|---|---|---|
| Encoder self-attention | previous encoder layer | previous encoder layer |
| Decoder self-attention (masked) | previous decoder layer | previous decoder layer |
| Cross-attention | previous decoder layer | **encoder output** |

Self-attention means Q, K, V all come from the same place. Cross-attention is
the asymmetric one: the query asks "what am I generating right now," and looks
it up in the encoder's representation of the source.

### Why the decoder's self-attention is masked

Paper's phrasing: prevent leftward information flow to preserve the
auto-regressive property.

Worked through concretely with "the cat sat on the mat":

- At **training**, the whole target sequence is fed in at once for speed. Without
  a mask, position 4 ("on") can attend to position 5 ("the") — which is exactly
  the token it is supposed to predict. It reads the answer instead of learning
  the language.
- At **inference**, the model only has the tokens it has already generated.
  Position 5 does not exist yet. The shortcut is gone and the model is useless.

The dangerous part: training loss looks excellent throughout. Low loss, smooth
convergence, no errors anywhere. The failure only appears at deployment.

Same silent-failure family as the Day 1 broadcasting bugs and the Day 2 operand
order bug — plausible output, no complaint, wrong result.

**To find:** the paper mentions setting masked values to −∞ before the softmax.
Worth working out why −∞ rather than 0. That is Day 11's implementation detail.

---

## 3.3 — Position-wise Feed-Forward Networks

Two linear transformations with a ReLU between them.

### Shape path (tracing one token)

```
input:            512 numbers
  ↓ Linear(512, 2048)
hidden:          2048 numbers
  ↓ ReLU              (negatives → 0, count unchanged)
hidden:          2048 numbers
  ↓ Linear(2048, 512)
output:           512 numbers
```

`d_ff = 2048` is a **4× expansion** over `d_model = 512`. Wide in the middle,
same width at both ends.

**Why it must return to 512:** the encoder stacks 6 of these layers. Layer 1's
output feeds layer 2, which expects 512. Same width in, same width out is what
makes stacking possible.

### "Applied to each position separately and identically"

**Separately** — the FFN processes one token at a time. Token 1's 512 numbers go
in, 512 come out, with no reference to any other token.

This is the contrast with attention. Attention *requires* looking at other
positions — that is its whole purpose. The FFN never does. Attention is where
positions talk to each other; the FFN is where each position is processed alone.

**Identically** — all tokens go through the *same* two Linear layers. One set of
weights, applied N times. Not N different sets.

### Implementation payoff

No loop needed. `nn.Linear` only touches the last dimension and leaves the
leading dimensions alone. Hand it `(B, S, 512)` and it returns `(B, S, 2048)`,
applying the same weights to every position independently.

"Separately and identically" comes for free from the shape semantics. This makes
the FFN the simplest module in the model: Linear → ReLU → Linear. Day 9.

---

## 3.4 — Embeddings and Softmax

### Weight sharing across three places

The same weight matrix is shared between:

1. The encoder input embedding
2. The decoder input embedding
3. The pre-softmax linear transformation (the final layer producing logits)

**Why this is sensible:** the embedding layer maps a token ID to a vector. The
final linear layer does the reverse — takes a vector and scores it against every
token in the vocabulary. Same relationship between tokens and vectors, read in
opposite directions, so one matrix serves both.

**Practical effect:** far fewer parameters. That matrix is
`vocab_size × d_model`. For a 37,000-token vocabulary at `d_model = 512` that is
roughly 19 million numbers — stored once instead of three times.

### The √d_model scaling

Embedding weights are multiplied by `√d_model` before use. At `d_model = 512`
that is a factor of about 22.6.

The paper states this without justification. The generally accepted reason:
embedding weights initialize small (roughly unit variance, so values around ±1),
while positional encodings are sines and cosines ranging over the full −1 to 1.
Added together unscaled, the positional signal would be comparable in magnitude
to the token identity itself. Scaling the embeddings up first makes token
content dominate and lets position act as a modifier.

**Note the contrast with `√d_k`:** the attention scaling has a clean derivation
from the variance argument. This one is an empirical choice the authors made
without explaining. File it as "do it because the paper does" and revisit if it
ever seems to matter.

---

## 3.5 — Positional Encoding

### Why it is needed at all

Attention computes pairwise dot products between tokens, and a dot product does
not care where its two vectors came from. Shuffle the token order and you get
the same set of scores, just relabeled.

So self-attention alone cannot distinguish "the cat sat" from "sat cat the".
Position has to be injected into the token representations *before* attention
sees them, because attention itself cannot recover it.

### The formula

Varies along two axes:

- **`pos`** — position in the sequence (token 0, 1, 2, …)
- **`i`** — which dimension of the vector. Even dimensions get sine, odd get
  cosine.

Every `(position, dimension)` pair gets its own value. The result is a matrix,
which is why plotting it as a heatmap on Day 8 shows structure rather than
noise.

### Why sinusoids rather than learned embeddings

Two reasons given:

**1. Relative position.** From the paper: for any fixed offset `k`, `PE(pos+k)`
can be represented as a linear function of `PE(pos)`.

This means "three tokens back" is the *same* linear transformation whether you
are at position 5 or position 500. The model can learn one operation instead of
memorizing the relationship separately at every position.

It is a property of sinusoids specifically — it falls out of the angle addition
formulas, where `sin(a+b)` and `cos(a+b)` expand into combinations of `sin(a)`,
`cos(a)`, `sin(b)`, `cos(b)`. A fixed offset means fixed coefficients.

**2. Extrapolation to longer sequences.** A learned position embedding table has
a fixed number of rows, so position 600 simply does not exist if training only
went to 512. A formula has no such limit — it can be evaluated at any position.

### The honest caveat

The paper also reports experimenting with learned positional embeddings and
finding nearly identical results. So the sinusoids are not magic. The
extrapolation property is the practical argument for them, not measured
performance.

---

## Still To Do

- [x] 3.4 — embeddings and softmax
- [x] 3.5 — positional encoding
- [x] Hand-drawn architecture diagram (must be drawn personally — it is the
      comprehension check).

---

## Visual Resources

Reading is not the most effective channel here, so supplementing with:

- **3Blue1Brown's transformer series** — two heavily animated videos, builds
  attention up geometrically rather than through notation. Best fit.
- **Karpathy, "Let's build GPT"** — two-hour live build. Watch, do not code
  along; the code needs to be mine.
- **Alammar, The Illustrated Transformer** — the diagram version of the paper.

Day 8's positional-encoding plotting exercise exists for this reason: the
sinusoid formula is opaque as an equation and immediately obvious as a heatmap.

---

## What Was Hard

Section 3.2.3 did not land on first read. What fixed it was not rereading — it
was tracing a concrete five-word sentence position by position until the failure
mode was visible.

**Generalizable:** when a section refuses to click, build a small example and
walk through it. The paper is written for people who already knew the field, and
its terseness hides mechanics that are obvious once instantiated.