import torch
import torch.nn as nn
from src.embedding import TokenEmbedding, PositionalEncoding
from src.layers import Encoder, Decoder

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, logits, target):
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            smooth_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            if self.ignore_index >= 0:
                smooth_dist[target == self.ignore_index] = 0
        loss = self.criterion(log_probs, smooth_dist)
        non_pad = (target != self.ignore_index).sum()
        return loss / non_pad


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, N, vocab_size, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.N = N
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.te = TokenEmbedding(vocab_size, d_model, dropout)
        self.pe = PositionalEncoding(d_model, max_len, dropout)
        self.e = Encoder(d_model, num_heads, d_ff, N, dropout)
        self.d = Decoder(d_model, num_heads, d_ff, N, dropout)
        self.tgt_te = TokenEmbedding(vocab_size, d_model, dropout)
        self.lp = nn.Linear(d_model, vocab_size, bias=False)
        self.tgt_te.embedding.weight = self.lp.weight

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        es = self.pe(self.te(src))
        encode = self.e(es, src_mask)
        embed_target = self.pe(self.tgt_te(tgt))
        decode = self.d(embed_target, encode, src_mask, tgt_mask)
        project = self.lp(decode)
        return project