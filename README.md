# Attention Is All You Need — From Scratch

A meticulous, zero-shortcut implementation of the original Transformer architecture in pure PyTorch, written entirely from scratch without AI assistance.

The goal of this project is to build a deep, first-principles understanding of the mathematical mechanics behind modern Large Language Models by avoiding high-level abstractions (`torch.nn.Transformer`) and implementing the tensor transformations manually.

**Status: Complete.** The full encoder-decoder Transformer is implemented, tested, and verified by overfitting a single batch to near-zero loss (3.6 → 0.005 in 500 steps).

---

## Mathematical Foundations

This repository implements the exact formulations introduced in the foundational paper *(Vaswani et al., 2017)*:

### 1. Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 2. Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 3. Sinusoidal Positional Encoding
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

---

## Proof of Correctness

The standard test for a from-scratch implementation: train on a single batch until the loss reaches near-zero. A model that cannot memorize two examples is broken somewhere; one that can has an intact gradient path through the entire architecture.

```
step   0, loss 3.6075
step  50, loss 0.1957
step 100, loss 0.0726
step 200, loss 0.0248
step 300, loss 0.0127
step 400, loss 0.0078
step 490, loss 0.0055
```

Task: sequence copying, batch size 2, sequence length 5, vocab size 20.

---

## Tensor Dimension Tracking Cheat Sheet

The core engineering challenge of this implementation is maintaining shape sanity across linear projections and matrix multiplications.

| Module / Operation | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| **Token Embedding** | `(B, S)` | `(B, S, d_model)` | `B`: Batch Size, `S`: Sequence Length |
| **Positional Encoding** | `(B, S, d_model)` | `(B, S, d_model)` | Additive element-wise operation |
| **Q, K, V Projections** | `(B, S, d_model)` | `(B, S, d_model)` | Linear layer per matrix |
| **Split into Heads** | `(B, S, d_model)` | `(B, H, S, d_k)` | `H`: Heads, $d_k = d_{\text{model}} / H$ |
| **Attention Weights ($QK^T$)** | `(B, H, S, d_k)` & `(B, H, S, d_k)` | `(B, H, S, S)` | Scaled and optionally masked |
| **MHA Output Concat** | `(B, H, S, d_k)` | `(B, S, d_model)` | Multiplied by output weight $W^O$ |
| **Feed-Forward Network** | `(B, S, d_model)` | `(B, S, d_model)` | Inner layer expands to $d_{\text{ff}} = 4 \times d_{\text{model}}$ |

---

## Project Architecture

The codebase is modularized into single-responsibility Python files:

- `src/attention.py` — Scaled dot-product attention, multi-head attention, head splitting and merging
- `src/embedding.py` — Token embedding with √d_model scaling, sinusoidal positional encoding
- `src/layers.py` — Feed-forward network, LayerNorm, encoder layer, encoder stack, decoder layer, decoder stack
- `src/model.py` — Full Transformer assembling all components

---

## Verification & Testing

Every module is verified by unit tests asserting correct shapes, and the attention function is verified against a hand-computed example derived on paper before any code was written.

### Running Unit Tests
```bash
pytest tests/
```

### Test Coverage
- `tests/test_attention.py` — scaled dot-product attention against hand-computed values; multi-head equivalence with h=1 and identity projections
- `tests/test_heads.py` — head split shape; round-trip split→merge returns original tensor
- `tests/test_embedding.py` — token embedding shape; positional encoding shape
- `tests/test_layers.py` — encoder layer, encoder stack, decoder layer shapes
- `tests/test_model.py` — full Transformer output shape

### Manual Sanity Checks
1. **Attention verification:** computed scaled dot-product attention by hand for a 2×2 example before implementation, then asserted the implementation matches.
2. **√d_k scaling:** empirically measured score variance at d_k = 4, 64, 512 — confirmed variance tracks d_k before scaling and equals 1 after.
3. **Look-ahead masking:** verified the attention weight matrix is lower-triangular when the causal mask is applied.
4. **Positional encoding:** plotted the PE matrix as a heatmap — confirmed the banded sinusoid pattern.
5. **Overfit one batch:** trained on two examples for 500 steps — loss reached 0.005.

---

## Learning Journal

Day-by-day notes documenting what was read, what was built, and what went wrong are in `notes/`. Every misconception, wrong prediction, and bug found is recorded there. The notes are the primary artifact of the learning process.

---

## References

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. arXiv preprint arXiv:1706.03762.