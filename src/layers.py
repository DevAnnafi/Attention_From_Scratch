import torch 
import torch.nn as nn
import math

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