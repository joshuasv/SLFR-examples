import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed; from sys import exit


from src.models.transformer_based.Encoder import EncoderLayer
from src.models.transformer_based.masked_batch_norm import MaskedBatchNorm1d

class Encoder(nn.Module):
    """Encoder based on: https://www.kaggle.com/code/irohith/aslfr-transformer"""
    def __init__(
            self,
            input_dim, 
            hid_dim,
            conv_kernel_size,
            n_conv_layers,
            n_layers, 
            n_heads, 
            pf_dim,
            dropout, 
        ):
        super().__init__()

        conv_layers = []
        in_channels = input_dim
        for _ in range(n_conv_layers):
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hid_dim,
                    kernel_size=conv_kernel_size,
                    padding='same'))
            in_channels = hid_dim
        self.conv_layers = nn.ModuleList(conv_layers)

        self.layers = nn.ModuleList(
            [EncoderLayer(
                hid_dim=hid_dim, 
                n_heads=n_heads, 
                pf_dim=pf_dim,  
                dropout=dropout, 
            ) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(name='scale', tensor=torch.FloatTensor([hid_dim]))

    def forward(self, seq, seq_mask):
        # seq = (batch_size, seq_len, input_dim)
        # seq_mask = (batch_size, 1, 1, seq_len)
        seq = seq.permute((0,2,1))
        # seq = (batch_size, input_dim, seq_len)
        for conv in self.conv_layers:
            seq = F.relu(conv(seq))
        # seq = (batch_size, hid_dim, seq_len)
        seq = seq.permute((0,2,1))
        # seq = (batch_size, seq_len, hid_dim)

        seq = self.dropout((seq * self.scale))
        # seq = (batch_size, seq_len, hid_dim)

        for layer in self.layers:
            seq = layer(seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim)

        return seq

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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.feeders.GASLFRDataset import GASLFRDataset
    from src.models.transformer_based.Seq2Seq import make_seq_mask
    from src.globals import SEQ_PAD_VALUE

    d = GASLFRDataset(
        split_csv_fpath='./data_gen/baseline/train_split.csv',
        max_phrase_len=45,
        prep_max_seq_len=384,
        prep_include_z=False,
        prep_include_vels=False,
        debug=False,
    )
    dl = DataLoader(
        dataset=d,
        batch_size=12,
        num_workers=0,
    )
    X, y = next(iter(dl))
    X_mask = make_seq_mask(X, seq_pad_idx=SEQ_PAD_VALUE)
    e = Encoder(
        input_dim=88,
        hid_dim=192,
        n_layers=5,
        n_heads=4,
        pf_dim=2048,
        dropout=0,
        max_seq_length=384,
    )
    e(X, X_mask)