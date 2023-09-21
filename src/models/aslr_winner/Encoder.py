import torch
import torch.nn as nn

from src.models.aslr_winner.Conv1dBlock import Conv1dBlock
from src.models.aslr_winner.masked_batch_norm import MaskedBatchNorm1d
from src.models.transformer_based.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from src.models.transformer_based.PositionWiseFeedForwardLayer import PositionWiseFeedForwardLayer


class Encoder(nn.Module):
    """Encoder based on: https://www.kaggle.com/competitions/asl-signs/discussion/406684"""
    def __init__(
            self,
            input_dim, 
            hid_dim, 
            n_layers, 
            n_heads, 
            pf_dim,
            dropout, 
        ):
        super().__init__()

        self.linear = nn.Linear(input_dim, hid_dim, bias=False)
        self.data_bn = MaskedBatchNorm1d(num_features=hid_dim, momentum=0.95)

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
        # seq = (batch_size, seq_len, seq_features)
        # seq_mask = (batch_size, 1, 1, seq_len)

        seq = self.linear(seq)
        # seq = (batch_size, seq_len, hid_dim)

        seq = seq.permute((0,2,1))
        # seq = (batch_size, hid_dim, seq_len)
        seq = self.data_bn(seq)
        seq = seq.permute((0,2,1))
        # seq = (batch_size, seq_len, hid_dim)

        for layer in self.layers:
            seq = layer(seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim)

        return seq


class EncoderLayer(nn.Module):

    def __init__(
            self,
            hid_dim, 
            n_heads, 
            pf_dim,  
            dropout, 
            kernel_size=17,
            dilation_rate=1,
            expand_ratio=2
        ):
        super().__init__()

        self.conv_block_1 = Conv1dBlock(
            in_channels=hid_dim,
            out_channels=hid_dim,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            drop_rate=dropout,
            expand_ratio=expand_ratio,
        )
        self.conv_block_2 = Conv1dBlock(
            in_channels=hid_dim,
            out_channels=hid_dim,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            drop_rate=dropout,
            expand_ratio=expand_ratio,
        )
        self.conv_block_3 = Conv1dBlock(
            in_channels=hid_dim,
            out_channels=hid_dim,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            drop_rate=dropout,
            expand_ratio=expand_ratio,
        )

        self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.poswise_ff = PositionWiseFeedForwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, seq_mask):
        # seq = (batch_size, seq_len, seq_features)
        # seq_mask = (batch_size, 1, 1, seq_len)

        seq = self.conv_block_1(seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim)
        seq = self.conv_block_2(seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim)
        seq = self.conv_block_3(seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim)

        seq_residual, _ = self.self_attn(seq, seq, seq, seq_mask)

        seq = self.self_attn_layer_norm(seq + self.dropout(seq_residual))
        # seq = (batch_size, seq_len, hid_dim)

        seq_residual = self.poswise_ff(seq)

        seq = self.ff_layer_norm(seq + self.dropout(seq_residual))
        # seq = (batch_size, seq_len, hid_dim)

        return seq
