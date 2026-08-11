import torch 
import torch.nn as nn
from src.embedding import TokenEmbedding, PositionalEncoding
from src.layers import Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, N, vocab_size, max_len):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.N = N
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.te = TokenEmbedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.e = Encoder(d_model, num_heads, d_ff, N)
        self.d = Decoder(d_model, num_heads, d_ff, N)
        self.tgt_te = TokenEmbedding(vocab_size, d_model)
        self.lp = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        es = self.pe(self.te(src))
        encode = self.e(es)
        embed_target = self.pe(self.tgt_te(tgt))
        decode = self.d(embed_target, encode, src_mask, tgt_mask)
        project = self.lp(decode)
        return project