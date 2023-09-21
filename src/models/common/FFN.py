import torch.nn as nn

from src.tools import activation_factory

class FFN(nn.Module):

    def __init__(self, input_dim, expand_dim, activation, dropout=0.0):
        super().__init__()
        activation_fn = activation_factory(activation)
        self.expand_linear = nn.Linear(input_dim, expand_dim)
        self.activation = activation_fn
        self.proj_linear = nn.Linear(expand_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        out = self.expand_linear(x)
        out = self.activation(out)
        out = self.proj_linear(out)
        out = self.dropout(out)
        out = self.lnorm(out + x)

        return out
        