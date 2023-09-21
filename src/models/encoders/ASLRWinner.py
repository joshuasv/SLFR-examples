import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math

from src.models.common.MaskedBatchNorm1d import MaskedBatchNorm1d
from src.models.common.MultiHeadAttentionLayer import MultiHeadAttentionLayer

class Encoder(nn.Module):

    def __init__(
            self,
            in_channels,
            hid_dim,
            emb_kernel_size,
            emb_dilation_rate,
            emb_drop_rate,
            emb_expand_ratio,
            emb_num_head,
            emb_num_layers,
            enc_num_heads,
            enc_expand_ratio,
            enc_drop_rate,
            enc_num_layers,
            out_dim,
        ):
        super().__init__()
        self.seq_emb = nn.ModuleList(
            [SeqEmbedding(
                in_channels=in_channels if i == 0 else hid_dim,
                hid_dim=hid_dim,
                kernel_size=emb_kernel_size,
                dilation_rate=emb_dilation_rate,
                drop_rate=emb_drop_rate,
                expand_ratio=emb_expand_ratio,
                num_head=emb_num_head
            ) for i in range(emb_num_layers)])
        self.features = nn.Linear(in_features=hid_dim, out_features=out_dim)

    def forward(self, seq, seq_mask):
        # seq = (batch, seq_len, kps*coords)
        # seq_mask = (batch, seq_len)

        for l in self.seq_emb:
            seq = l(seq, seq_mask)
        
        # for l in self.encoder:
        #     seq = l(seq, seq_mask)
        seq = self.features(seq)

        return seq


class SeqEmbedding(nn.Module):

    def __init__(
            self,
            in_channels, 
            hid_dim, 
            kernel_size, 
            dilation_rate, 
            drop_rate, 
            expand_ratio,
            num_head,
        ):
        super().__init__()
        
        self.conv1 = Conv1dBlock(
            in_channels=in_channels, 
            out_channels=hid_dim,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            drop_rate=drop_rate,
            expand_ratio=expand_ratio
        )
        self.conv2 = Conv1dBlock(
            in_channels=hid_dim, 
            out_channels=hid_dim,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            drop_rate=drop_rate,
            expand_ratio=expand_ratio
        )
        self.conv3 = Conv1dBlock(
            in_channels=hid_dim, 
            out_channels=hid_dim,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            drop_rate=drop_rate,
            expand_ratio=expand_ratio
        )
        self.tformer = TransformerBlock(
            embed_dim=hid_dim,
            num_heads=num_head,
            expand=expand_ratio,
            rate=drop_rate
        )

    def forward(self, seq, seq_mask):
        # seq = (batch, seq_len, kps*coords)
        # seq_mask = (batch, seq_len)

        seq = self.conv1(seq, seq_mask)
        seq = self.conv2(seq, seq_mask)
        seq = self.conv3(seq, seq_mask)
        
        seq = self.tformer(seq, seq_mask)
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
        # seq = (batch, seq_len, features)
        # seq_mask = (batch, seq_len); true if pad

        skip = seq
        seq = F.relu(self.linear_expand(seq))
        # seq = (batch, seq_len, features*expand)

        seq = self.dw_conv(seq, seq_mask)
        # seq = (batch_size, seq_len, features*expand)

        seq = seq.permute((0,2,1))
        if seq_mask is not None:
            seq_mask = seq_mask.unsqueeze(1)
        # seq = self.bn(seq, input_mask=seq_mask)
        seq = self.bn(seq, input_mask=None)
        seq = seq.permute((0,2,1))
        if seq_mask is not None:
            seq_mask = seq_mask.squeeze(1)
        # seq = (batch_size, seq_len, features*expand)

        # seq = self.eca(seq, seq_mask)
        seq = self.eca(seq, None)
        # seq = (batch_size, seq_len, features*expand)
        
        seq = self.lienar_proj(seq)

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
        # seq = (batch_size  seq_len, features)
        # seq_mask = (batch, seq_len)

        seq = F.pad(seq, self.p3d)
        # seq = (batch_size  seq_len+padding, features)
        seq = torch.permute(seq, (0, 2, 1))
        # seq = (batch_size, features, seq_len+padding)
        seq = self.dw_conv(seq)
        # seq = (batch_size, features, seq_len)
        seq = torch.permute(seq, (0, 2, 1))
        # seq = (batch_size, seq_len, features)

        # if seq_mask is not None:
        #     seq_mask = seq_mask.squeeze().unsqueeze(2)
        #     # seq_mask = (batch_size, seq_len, 1)
        #     # seq = seq.masked_fill(seq_mask==0, -1e10)
        #     seq = seq * seq_mask

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
        # seq = (batch_size, seq_len, features)
        # seq_mask = (batch, seq_len); true if pad
        skip = seq
        
        if seq_mask is not None:
            seq = seq.masked_fill(seq_mask.unsqueeze(2), torch.nan)
            seq = seq.nanmean(dim=1, keepdim=True)
        else:
            seq = seq.mean(dim=1, keepdim=True)
        # seq = (batch_size, 1, features)

        seq = self.conv(seq)
        # seq = (batch_size, 1, features)

        seq = self.sigmoid(seq)
        seq = seq * skip
        if seq_mask is not None:
            seq = seq * torch.logical_not(seq_mask.unsqueeze(2)) # 0 if pad

        # seq = (batch_size, seq_len, features)
        return seq


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            expand,
            rate,
        ):
        super().__init__()

        feed_forward_dim = embed_dim * expand
        # self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn = MultiHeadAttentionLayer(hid_dim=embed_dim, n_heads=num_heads, dropout=0.0)
        ffn = [
            nn.Linear(in_features=embed_dim, out_features=feed_forward_dim),
            nn.ReLU(),
            nn.Linear(in_features=feed_forward_dim, out_features=embed_dim),
        ]
        self.ffn = nn.Sequential(*ffn)
        self.lnorm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.lnorm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout = nn.Dropout(rate)

    def forward(self, seq, seq_mask):
        # seq = (batch, seq_len, hid_dim)
        # seq_mask = (batch, seq_len)
        
        skip = seq

        seq, _ = self.attn(seq, seq, seq, mask=seq_mask)
        # seq = (batch, seq_len, hid_dim)
        
        seq = self.dropout(seq)
        seq = seq + skip
        seq = self.lnorm1(seq)
        # seq = (batch, seq_len, hid_dim)

        skip = seq
        seq = self.ffn(seq)
        # seq = (batch, seq_len, hid_dim)

        seq = self.dropout(seq)
        seq = seq + skip
        seq = self.lnorm2(seq)
        # seq = (batch, seq_len, hid_dim)

        return seq
