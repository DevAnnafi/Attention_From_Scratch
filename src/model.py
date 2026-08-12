import torch
import torch.nn as nn
import math
from src.embedding import TokenEmbedding, PositionalEncoding
from src.layers import Encoder, Decoder


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss — section 5.4.

    Smoothing=0.1: model is penalized for overconfidence.
    Target distribution: (1-eps) on correct class, eps/(V-2) on non-pad classes.

    Fix: <pad> column receives zero smoothing mass at every real position,
    and pad rows are excluded entirely from both numerator and denominator.

    Works on 3D input (B, T, V) — flattens to (B*T, V) internally.
    """
    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, T, V) — raw model output
            target: (B, T)   — gold token IDs (long)
        """
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        target = target.reshape(B * T)

        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            # Distribute smoothing mass over (V - 2) classes — exclude correct
            # class AND <pad>. This prevents the model from learning to emit padding.
            smooth = self.smoothing / (V - 2)
            dist = torch.full_like(log_probs, smooth)
            dist[:, self.pad_idx] = 0.0                          # zero pad column
            dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            dist[target == self.pad_idx] = 0.0                   # zero pad rows

        loss = self.criterion(log_probs, dist)
        non_pad = (target != self.pad_idx).sum().clamp(min=1)
        return loss / non_pad


class Transformer(nn.Module):
    """Full encoder-decoder Transformer (Vaswani et al., 2017).

    Weight tying (section 3.4):
      target embedding <-> output projection share one matrix.
      Source embedding is separate (enables different src/tgt vocabs).

    Initialization:
      - All linear/projection layers: Xavier uniform (standard for transformers)
      - Embedding matrix: N(0, d_model^-0.5) — NOT Xavier, which is 6x too
        small at WMT14 scale (V=37k, d_model=512) and causes slow warmup.

    Args:
        d_model:    model dimension
        num_heads:  attention heads (must divide d_model)
        d_ff:       feed-forward inner dimension (paper: 4 * d_model)
        N:          number of encoder and decoder layers
        vocab_size: shared vocabulary size
        max_len:    maximum sequence length for positional encoding
        dropout:    dropout probability (paper: 0.1)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        N: int,
        vocab_size: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_embed = TokenEmbedding(vocab_size, d_model, dropout=0.0)
        self.tgt_embed = TokenEmbedding(vocab_size, d_model, dropout=0.0)
        self.pe = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = Encoder(d_model, num_heads, d_ff, N, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, N, dropout)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying — target embedding and output projection share weights
        self.tgt_embed.embedding.weight = self.projection.weight

        self._init_weights(d_model)

    def _init_weights(self, d_model: int):
        """Xavier uniform for all non-embedding params; N(0, d_model^-0.5) for embeddings."""
        for name, p in self.named_parameters():
            if p is self.projection.weight:
                # Tied weight — initialized once via tgt_embed below
                continue
            if 'embedding' in name:
                # Embedding rows: N(0, d_model^-0.5)
                # At WMT14 scale this is ~0.044; Xavier gives ~0.007 (6x too small)
                nn.init.normal_(p, mean=0.0, std=d_model ** -0.5)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask=None) -> torch.Tensor:
        return self.encoder(self.pe(self.src_embed(src)), src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask=None,
        tgt_mask=None,
    ) -> torch.Tensor:
        return self.decoder(self.pe(self.tgt_embed(tgt)), memory, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask=None,
        tgt_mask=None,
    ) -> torch.Tensor:
        """
        Args:
            src: (B, S_src) token IDs
            tgt: (B, S_tgt) token IDs — must be shifted right before calling
        Returns:
            (B, S_tgt, vocab_size) logits
        """
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.projection(out)