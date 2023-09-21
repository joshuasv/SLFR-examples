import torch.nn as nn
import torch.nn.functional as F

from src.models.aslr_winner.CausalDwConv1d import CausalDwConv1d
from src.models.aslr_winner.masked_batch_norm import MaskedBatchNorm1d
from src.models.aslr_winner.ECA import ECA

from IPython import embed; from sys import exit

class Conv1dBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation_rate=1,
            drop_rate=0.0,
            expand_ratio=2,
            se_ratio=0.25, # Can it be squeeze and excitation? https://arxiv.org/abs/1709.01507
        ):
        super().__init__()

        self.add_skip = in_channels == out_channels

        channels_expand = in_channels * expand_ratio
        self.linear_expand = nn.Linear(in_channels, channels_expand)
        self.dw_conv = CausalDwConv1d(
            channels=channels_expand,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False
        )
        self.bn = MaskedBatchNorm1d(channels_expand, momentum=0.95)
        self.eca = ECA()
        self.lienar_proj = nn.Linear(channels_expand, out_channels)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, seq, seq_mask):
        # seq = (batch_size, seq_len, hid_dim)
        # seq_mask = (batch_size, 1, 1, seq_len)
        skip = seq
        seq = self.linear_expand(seq)
        seq = F.relu(seq)
        # seq = (batch_size, seq_len, hid_dim*expand_ratio)
        seq = self.dw_conv(seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim*expand_ratio)
        seq = seq.permute((0,2,1))
        # seq = (batch_size, hid_dim*expand_ratio, seq_len)
        seq = self.bn(seq)
        seq = seq.permute((0,2,1))
        # seq = (batch_size, seq_len, hid_dim*expand_ratio)
        seq = self.eca(seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim*expand_ratio)
        seq = self.lienar_proj(seq)
        # seq = (batch_size, seq_len, hid_dim)
        seq = self.dropout(seq)
        if self.add_skip:
            seq = seq + skip

        # seq = (batch_size, seq_len, hid_dim)
        return seq