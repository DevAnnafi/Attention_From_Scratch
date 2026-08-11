import torch
import pytest
from src.masks import create_padding_mask, create_causal_mask, create_masks


def test_padding_mask_shape():
    seq = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])  # B=2, S=4
    mask = create_padding_mask(seq, pad_idx=0)
    assert mask.shape == (2, 1, 1, 4)


def test_padding_mask_values():
    seq = torch.tensor([[1, 2, 0, 0]])
    mask = create_padding_mask(seq, pad_idx=0)
    expected = torch.tensor([[[[1., 1., 0., 0.]]]])
    assert torch.equal(mask, expected), f"Got {mask}"


def test_causal_mask_shape():
    mask = create_causal_mask(5)
    assert mask.shape == (1, 1, 5, 5)


def test_causal_mask_is_lower_triangular():
    mask = create_causal_mask(5).squeeze()
    upper = torch.triu(mask, diagonal=1)
    assert torch.all(upper == 0), f"Upper triangle not zero:\n{mask}"
    diag = torch.diagonal(mask)
    assert torch.all(diag == 1), f"Diagonal not ones:\n{mask}"


def test_create_masks_shapes():
    src = torch.randint(0, 10, (2, 6))
    tgt = torch.randint(0, 10, (2, 5))
    src_mask, tgt_mask = create_masks(src, tgt, pad_idx=0)
    assert src_mask.shape == (2, 1, 1, 6)
    assert tgt_mask.shape == (2, 1, 5, 5)


def test_combined_tgt_mask_blocks_future_and_padding():
    """Decoder mask must block both future positions and padding."""
    # tgt: [BOS, tok, PAD, PAD] — position 2 and 3 are padding
    tgt = torch.tensor([[1, 5, 0, 0]])
    _, tgt_mask = create_masks(tgt, tgt, pad_idx=0)
    mask = tgt_mask.squeeze()  # (4, 4)

    # Upper triangle should be 0 (causal)
    upper = torch.triu(mask, diagonal=1)
    assert torch.all(upper == 0), f"Causal part failed:\n{mask}"

    # Columns 2 and 3 should be 0 (padding mask)
    assert torch.all(mask[:, 2:] == 0), f"Padding part failed:\n{mask}"