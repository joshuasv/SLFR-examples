import torch
import torch.nn as nn


class EarlyDropout(nn.Module):

    def __init__(self, rate, end_epoch=0):
        super().__init__()
        self.rate = rate
        self.end_epoch = end_epoch
        self.dropout = nn.Dropout(p=rate)
        self.register_buffer('train_counter', torch.tensor(0, dtype=torch.int32))
        self._num_steps_per_epoch = None

    @property
    def end_step(self):
        return self.num_steps_per_epoch * self.end_epoch

    @property
    def num_steps_per_epoch(self):
        return self._num_steps_per_epoch
    
    @num_steps_per_epoch.setter
    def num_steps_per_epoch(self, value):
        self._num_steps_per_epoch = value

    def forward(self, x):
        if self.rate == 0. or not self.training:
            return x
        
        if self.train_counter < self.end_step:
            x = self.dropout(x)
            self.train_counter += 1

        return x



