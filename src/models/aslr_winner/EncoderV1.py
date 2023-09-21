import torch
import torch.nn as nn

from src.models.aslr_winner.Conv1dBlock import Conv1dBlock
from src.models.aslr_winner.masked_batch_norm import MaskedBatchNorm1d
from src.models.transformer_based.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from src.models.transformer_based.PositionWiseFeedForwardLayer import PositionWiseFeedForwardLayer

from IPython import embed; from sys import exit

class Encoder(nn.Module):
    """Encoder based on: https://www.kaggle.com/competitions/asl-signs/discussion/406684"""
    def __init__(
            self,
            input_dim, 
            hid_dim,
            n_layers,
            n_heads, 
            dropout, 
            kernel_size=17,
            expand_ratio=2
        ):
        super().__init__()
        self.linear = nn.Linear(input_dim, hid_dim, bias=False)
        self.data_bn = MaskedBatchNorm1d(num_features=hid_dim, momentum=0.95)
        self.seq_emb = nn.ModuleList(
            [EncoderLayer(
                hid_dim=hid_dim,
                n_heads=n_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                expand_ratio=expand_ratio
            ) for _ in range(n_layers)]
        )
        # self.top_linear = nn.Linear(hid_dim, hid_dim*2)


    def forward(self, seq, seq_mask=None):
        # seq = (batch_size, seq_len, seq_features)
        # seq_mask = (batch_size, 1, 1, seq_len)
        seq = self.linear(seq)
        # seq = (batch_size, seq_len, hid_dim)
        seq = seq.permute((0,2,1))
        seq_mask = seq_mask.squeeze(1)
        # seq = (batch_size, hid_dim, seq_len)
        # seq_mask = (batch_size, 1, seq_len)
        seq = self.data_bn(seq, seq_mask)
        seq = seq.permute((0,2,1))
        seq_mask = seq_mask.unsqueeze(1)
        # seq = (batch_size, seq_len, hid_dim)
        # seq_mask = (batch_size, 1, 1, seq_len)
        for l in self.seq_emb:
            seq = l(seq, seq_mask)
        # seq = self.top_linear(seq)

        # seq = (batch_size, seq_len, hid_dim*2)
        return seq


class EncoderLayer(nn.Module):
    
    def __init__(
            self,
            hid_dim, 
            n_heads, 
            dropout, 
            kernel_size=17,
            dilation_rate=1,
            expand_ratio=2
        ):
        super().__init__()
        pf_dim = hid_dim * expand_ratio

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
        self.tformer_block = TransformerBlock(
            hid_dim=hid_dim,
            n_heads=n_heads,
            dropout=dropout,
            pf_dim=pf_dim
        )

    def forward(self, seq, seq_mask=None):
        # seq = (batch_size, seq_len, hid_dim)
        # seq_mask = (batch_size, 1, 1, seq_len)
        seq = self.conv_block_1(seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim)
        seq = self.conv_block_2(seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim)
        seq = self.conv_block_3(seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim)
        seq = self.tformer_block(seq, seq_mask)

        # seq = (batch_size, seq_len, hid_dim)
        return seq
    

class TransformerBlock(nn.Module):

    def __init__(
            self,
            hid_dim,
            n_heads,
            dropout,
            pf_dim,
        ):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(num_features=hid_dim, momentum=0.95)
        self.lnorm1 = nn.LayerNorm(hid_dim)
        self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(num_features=hid_dim, momentum=0.95)
        self.lnorm2 = nn.LayerNorm(hid_dim)
        self.poswise_ff = PositionWiseFeedForwardLayer(hid_dim, pf_dim, dropout)
        self.act = nn.ReLU()

    def forward(self, seq, seq_mask):
        # seq = (batch_size, seq_len, hid_dim)
        # seq_mask = (batch_size, 1, 1, seq_len)
        skip = seq
        seq = seq.permute(0,2,1)
        seq = self.bn1(seq)
        seq = seq.permute(0,2,1)
        # seq = (batch_size, seq_len, hid_dim)
        seq = self.lnorm1(seq)
        seq, _ = self.self_attn(seq, seq, seq, seq_mask)
        # seq = (batch_size, seq_len, hid_dim)
        seq = self.dropout(seq)
        seq = skip + seq

        skip = seq
        seq = seq.permute(0,2,1)
        seq = self.bn2(seq)
        seq = seq.permute(0,2,1)
        seq = self.lnorm1(seq)
        seq = self.poswise_ff(seq)
        seq = skip + seq

        # seq = (batch_size, seq_len, hid_dim)
        return seq


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.feeders.GASLFRDataset import GASLFRDataset
    from src.models.transformer_based.Seq2Seq import make_seq_mask
    from src.globals import SEQ_PAD_VALUE
    from IPython import embed; from sys import exit

    m = Encoder(
        input_dim=88,
        hid_dim=256,
        n_layers=2,
        n_heads=4,
        dropout=0.2, 
        kernel_size=17,
        expand_ratio=2
    )
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
        num_workers=0
    )
    X, y = next(iter(dl))
    X_mask = make_seq_mask(X, seq_pad_idx=SEQ_PAD_VALUE)
    o = m(X, X_mask)