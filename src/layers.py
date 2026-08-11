import torch 
import torch.nn as nn
import math
from src.attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(
            self.relu(
                self.linear1(x)
            )
        )

class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        x1 = self.norm1(x + self.attention(x,x,x))
        x2 = self.norm2(x1 + self.ffn(x1))
        return x2

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, N):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.N = N
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(N)])

    def forward(self, x):
        result = x
        for layer in self.layers:
            result = layer(result)
        return result

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ma = MultiHeadAttention(d_model, num_heads)
        self.ca = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        masked_attention = self.ma(x, x, x, tgt_mask)
        x = self.norm1(x + masked_attention)
        cross_attention = self.ca(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + cross_attention)
        feed_forward = self.ffn(x)
        x = self.norm3(x + feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, N):
         super().__init__()
         self.d_model = d_model
         self.num_heads = num_heads
         self.d_ff = d_ff
         self.N = N
         self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(N)])
         
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        result = x
        for layer in self.layers:
            result = layer(result, encoder_output, src_mask, tgt_mask)
        return result




        
        