import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tools import activation_factory, initialization_factory
from src.models.common.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from src.models.common.DropPathLayer import DropPath

class Encoder(nn.Module):

    def __init__(
            self, 
            in_channels, 
            num_hid, 
            num_head, 
            num_feed_forward, 
            num_layers, 
            learn_pos_enc=False, 
            max_seq_len=None,
            seq_emb_dropout=0.0,
            seq_emb_activation='relu',
            seq_emb_init='he',
            seq_emb_learn_nan_emb=False,
            tformer_activation='relu',
            tformer_init='he',
            tformer_dropout=0.0, 
            tformer_attn_dropout=0.0,
            tformer_ffn_dropout=0.0,
            tformer_first_drop_path=0.0,
            tformer_second_drop_path=0.0
        ):
        super().__init__()
        self.learn_pos_enc = learn_pos_enc
        self.seq_emb = SeqEmbedding(
            in_channels=in_channels, 
            num_hid=num_hid, 
            activation=seq_emb_activation,
            init=seq_emb_init,
            learn_nan_emb=seq_emb_learn_nan_emb
        )
        self.seq_emb_dropout = nn.Dropout(seq_emb_dropout)
        self.seq_emb_lnorm = nn.LayerNorm(normalized_shape=num_hid)

        self.encoder = nn.ModuleList(
            [TransformerEncoder(
                embed_dim=num_hid, 
                num_heads=num_head, 
                feed_forward_dim=num_feed_forward, 
                activation=tformer_activation,
                init=tformer_init,
                dropout=tformer_dropout,
                attn_dropout=tformer_attn_dropout,
                ffn_dropout=tformer_ffn_dropout,
                first_drop_path=tformer_first_drop_path,
                second_drop_path=tformer_second_drop_path
            ) for _ in range(num_layers)]
        )
        if learn_pos_enc:
            self.pos_emb = nn.parameter.Parameter(
                data=torch.zeros((max_seq_len, num_hid), dtype=torch.float32),
                requires_grad=True
            )

    def forward(self, seq, seq_mask):
        # seq = (batch, seq_len, features)
        seq = seq.permute(0,2,1)
        seq = self.seq_emb(seq)
        seq = seq.permute(0,2,1)
        if self.learn_pos_enc:
            seq = seq + self.pos_emb

        seq = self.seq_emb_lnorm(seq)
        seq = self.seq_emb_dropout(seq)

        for l in self.encoder:
            seq = l(seq, seq_mask)

        return seq
    

class SeqEmbedding(nn.Module):

    def __init__(self, in_channels, num_hid, activation='relu', init='he', learn_nan_emb=False):
        super().__init__()
        self.learn_nan_emb = learn_nan_emb
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=num_hid, kernel_size=11, padding='same')
        self.conv2 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, padding='same')
        self.conv3 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, padding='same')
        self.activation_fn = activation_factory(activation)

        if learn_nan_emb:
            self.nan_emb = nn.parameter.Parameter(
                data=torch.zeros((1, num_hid, 1), dtype=torch.float32),
                requires_grad=True
            )
            self.num_hid = num_hid

    def forward(self, seq):
        if self.learn_nan_emb:
            mask = (seq==0.0).all(1, keepdim=True).repeat(1, self.num_hid, 1) # REVISAR, porque la reducción? Es en todos los elmenetos nan
        # seq = (batch, channels, len)
        seq = self.activation_fn(self.conv1(seq))
        seq = self.activation_fn(self.conv2(seq))
        seq = self.activation_fn(self.conv3(seq))
        # seq = (batch, num_hid, len)
        if self.learn_nan_emb:
            seq = torch.where(mask, self.nan_emb, seq)
        return seq
    

class TransformerEncoder(nn.Module):

    def __init__(
            self,
            embed_dim, 
            num_heads, 
            feed_forward_dim, 
            activation='relu',
            init='he',
            dropout=0.0, 
            attn_dropout=0.0, 
            ffn_dropout=0.0, 
            first_drop_path=0.0, 
            second_drop_path=0.0
        ):
        super().__init__()
        activation_fn = activation_factory(activation)
        self.attn = MultiHeadAttentionLayer(hid_dim=embed_dim, n_heads=num_heads, dropout=attn_dropout)
        ffn = [
            nn.Linear(in_features=embed_dim, out_features=feed_forward_dim),
            activation_fn,
            nn.Dropout(ffn_dropout),
            nn.Linear(in_features=feed_forward_dim, out_features=embed_dim),
        ]
        self.ffn = nn.Sequential(*ffn)

        self.lnorm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.lnorm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.drop_path1 = DropPath(drop_prob=first_drop_path)
        self.drop_path2 = DropPath(drop_prob=second_drop_path)

    def forward(self, seq, seq_mask):
        # seq = (batch, num_hid, len)
        # seq = (batch, len, num_hid)
        skip = seq
        # seq, _ = self.attn(seq, seq, seq, key_padding_mask=seq_mask, need_weights=False)
        seq, _ = self.attn(seq, seq, seq, mask=seq_mask)
        # seq = (batch, len, num_hid)
        seq = self.dropout(seq)
        seq = self.drop_path1(seq) + skip
        seq = self.lnorm1(seq)

        skip = seq
        seq = self.ffn(seq)
        seq = self.dropout(seq)
        seq = self.drop_path2(seq) + skip
        seq = self.lnorm2(seq)
        # seq = (batch, num_hid, len)

        return seq