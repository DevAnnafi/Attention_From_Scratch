# Attention Is All You Need — From Scratch

A meticulous, zero-shortcut implementation of the original Transformer architecture in pure PyTorch, written entirely from scratch without AI assistance. 

The goal of this project is to build a deep, first-principles understanding of the mathematical mechanics behind modern Large Language Models by avoiding high-level abstractions (`torch.nn.Transformer`) and implementing the tensor transformations manually.

##  Mathematical Foundations

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

## Tensor Dimension Tracking Cheat Sheet

The core engineering challenge of this implementation is maintaining shape sanity across linear projections and matrix multiplications. 

| Module / Operation | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| **Token Embedding** | `(B, S)` | `(B, S, d_model)` | `B`: Batch Size, `S`: Sequence Length |
| **Positional Encoding** | `(B, S, d_model)` | `(B, S, d_model)` | Additive element-wise operation |
| **Q, K, V Projections** | `(B, S, d_model)` | `(B, S, d_model)` | Linear layer per matrix |
| **Split into Heads** | `(B, S, d_model)` | `(B, H, S, d_k)` | `H`: Heads, $d_k = d_{\text{model}} / H$ |
| **Attention Weights ($QK^T$)**| `(B, H, S, d_k)` & `(B, H, S, d_k)` | `(B, H, S, S)` | Scaled and optionally masked |
| **MHA Output Concat** | `(B, H, S, d_k)` | `(B, S, d_model)` | Multiplied by output weight $W^O$ |
| **Feed-Forward Network** | `(B, S, d_model)` | `(B, S, d_model)` | Inner layer expands to $d_{\text{ff}} = 4 \times d_{\text{model}}$ |

---

## Project Architecture

The codebase is modularized cleanly into single-responsibility Python files:
* `src/embedding.py`: Handles token embedding lookup and fixed sinusoidal absolute position injection.
* `src/attention.py`: Core logic for scaled dot-product matrix multiplication and multi-head parallel tracking.
* `src/layers.py`: Implements Feed-Forward sub-layers, Layer Normalization, and structural residual wrappers.
* `src/model.py`: Orchestrates Encoder and Decoder block stacking into a single end-to-end network.

---

## Verification & Testing

Since this project avoids automated AI synthesis, verification relies heavily on structural unit tests to assert correct shape configurations and forward pass execution.

### Running Unit Tests
To run the automated suite testing tensor shapes, masking operations, and gradient paths, run:
```bash
pytest tests/
```

### Manual Sanity Checks
1. **Look-Ahead Masking Verification:** Asserts that token indices at position $t$ have zero attention weight assigned to any token index $> t$ in the causal decoder block.
2. **Residual Conservation:** Asserts that adding input gradients directly to layer outputs (`x + Sublayer(x)`) scales without encountering immediate vanishing or exploding gradients.

##  References
* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. arXiv preprint arXiv:1706.03762.

