"""
Mask construction for the Transformer.

Convention: 1 = attend, 0 = block.
`attention()` does `scores.masked_fill(mask == 0, _MASK_VALUE)`.

Two mask types:

  padding mask  — keeps attention away from <pad> tokens in the source.
                  Shape: (B, 1, 1, S) broadcasts over (B, H, S_q, S_k).

  causal mask   — keeps the decoder from seeing future positions.
                  Shape: (1, 1, T, T) broadcasts over (B, H, T, T).

  combined mask — elementwise AND of both; used for decoder self-attention
                  when the target sequence may also contain padding.
"""

import torch


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Build a key-padding mask.

    Args:
        seq:     (B, S) integer token IDs
        pad_idx: token ID used for <pad>

    Returns:
        (B, 1, 1, S) float tensor — 1.0 for real tokens, 0.0 for padding.
    """
    mask = (seq != pad_idx).float()          # (B, S)  1=keep, 0=pad
    return mask.unsqueeze(1).unsqueeze(2)    # (B, 1, 1, S)


def create_causal_mask(size: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Build a causal (look-ahead) mask.

    Position t may attend to 0..t only.

    Args:
        size:   target sequence length T
        device: tensor device

    Returns:
        (1, 1, T, T) float tensor — 1.0 on and below the diagonal, 0.0 above.
    """
    mask = torch.tril(torch.ones(size, size, device=device))  # lower triangular
    return mask.unsqueeze(0).unsqueeze(0)                     # (1, 1, T, T)


def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0):
    """
    Convenience function — returns both masks for a training step.

    Args:
        src:     (B, S_src) source token IDs
        tgt:     (B, S_tgt) target token IDs (already shifted right)
        pad_idx: padding token ID

    Returns:
        src_mask: (B, 1, 1, S_src) for encoder self-attention and cross-attention
        tgt_mask: (B, 1, T, T)     causal AND padding mask for decoder self-attention

    Usage:
        src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)
        out = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(out, tgt_labels)
    """
    src_mask = create_padding_mask(src, pad_idx)                        # (B,1,1,S_src)

    T = tgt.shape[1]
    causal = create_causal_mask(T, device=tgt.device)                  # (1,1,T,T)
    tgt_pad = create_padding_mask(tgt, pad_idx)                        # (B,1,1,T)
    tgt_mask = causal * tgt_pad                                        # (B,1,T,T) AND

    return src_mask, tgt_mask