import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDwConv1d(nn.Module):
    # https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch
    def __init__(
        self,
        channels,
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer='glorot_uniform'
    ):
        super().__init__()
        pad = dilation_rate * (kernel_size - 1)
        self.p3d = (0, 0, pad, 0, 0, 0)
        self.dw_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding='valid',
            dilation=dilation_rate,
            groups=channels,
            bias=use_bias
        )
        
    def forward(self, seq, seq_mask=None):
        # seq = (batch_size  seq_len, hid_dim*expand_ratio)
        # seq_mask = (batch_size, 1, 1, seq_len)
        seq = F.pad(seq, self.p3d)
        # seq = (batch_size  seq_len+padding, hid_dim*expand_ratio)
        seq = torch.permute(seq, (0, 2, 1))
        # seq = (batch_size, hid_dim*expand_ratio, seq_len+padding)
        seq = self.dw_conv(seq)
        # seq = (batch_size, hid_dim*expand_ratio, seq_len)
        seq = torch.permute(seq, (0, 2, 1))
        # seq = (batch_size, seq_len, hid_dim*expand_ratio)
        if seq_mask is not None:
            seq_mask = seq_mask.squeeze().unsqueeze(2)
            # seq_mask = (batch_size, seq_len, 1)
            # seq = seq.masked_fill(seq_mask==0, -1e10)
            seq = seq * seq_mask

        # seq = (batch_size, seq_len, hid_dim*expand_ratio)
        return seq
