import torch
import torch.nn as nn

class ECA(nn.Module):
    """Could it be Efficient Channel Attention?
    
    https://arxiv.org/pdf/1910.03151.pdf
    """
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, seq, seq_mask=None):
        # seq = (batch_size, seq_len, hid_dim*expand_ratio)
        # seq_mask = (batch_size, 1, 1, seq_len)
        skip = seq
        seq = seq.permute((0,2,1))
        # seq = (batch_size, hid_dim*expand_ratio, seq_len)
        if seq_mask is not None:
            seq_mask = seq_mask.squeeze(1)
            # seq_mask = (batch_size, 1, seq_len)
            seq = seq.masked_fill(seq_mask == 0, torch.nan)
            seq = seq.nanmean(dim=2, keepdim=True)
        else:
            seq = seq.mean(dim=2, keepdim=True)
        # seq = (batch_size, hid_dim*expand_ratio, 1)
        seq = seq.permute((0,2,1))
        # seq = (batch_size, 1, hid_dim*expand_ratio)
        seq = self.conv(seq)
        # seq = (batch_size, 1, hid_dim*expand_ratio)
        seq = self.sigmoid(seq)
        if seq_mask is not None:
            seq_mask = seq_mask.permute((0,2,1))
            # seq_mask = (batch_size, seq_len, 1)
            skip = skip * seq_mask
        seq = seq * skip

        # seq = (batch_size, seq_len, hid_dim*expand_ratio)
        return seq