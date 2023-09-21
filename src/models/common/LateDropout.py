import torch
import torch.nn as nn


class LateDropout(nn.Module):
    """https://arxiv.org/pdf/2303.01500.pdf"""
    def __init__(self, rate, start_epoch=0):
        super().__init__()
        self.rate = rate
        self.start_epoch = start_epoch
        self.dropout = nn.Dropout(p=rate)
        self.register_buffer('train_counter', torch.tensor(0, dtype=torch.int32))

    def forward(self, x):
        if self.rate == 0. or not self.training:
            return x
        
        if self.train_counter >= self.start_epoch:
            x = self.dropout(x)
        else:
            self.train_counter += 1

        return x



