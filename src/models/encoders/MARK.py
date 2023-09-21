import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tools import activation_factory, initialization_factory
from src.models.common.DropPathLayer import DropPath
import numpy as np

from IPython import embed; from sys import exit

class Encoder(nn.Module):

    def __init__(
            self, 
            in_channels, 
            num_hid, 
            num_head, 
            num_feed_forward, 
            num_layers, 
            dropout=0.0, 
            learn_pos_enc=False, 
            max_seq_len=None,
            seq_emb_activation='relu',
            seq_emb_init='he',
            seq_emb_learn_nan_emb=False,
            tformer_activation='relu',
            tformer_init='he',
            tformer_mha_dropout=0.0,
            tformer_ffn_dropout=0.0,
            tformer_first_drop_path=0.0,
            tformer_second_drop_path=0.0
        ):
        super().__init__()
        self.num_hid = num_hid
        self.learn_pos_enc = learn_pos_enc
        self.seq_emb = SeqEmbedding(
            in_channels=in_channels, 
            num_hid=num_hid, 
            max_seq_len=max_seq_len
        )
        self.encoder = nn.ModuleList(
            [TransformerEncoder(
                embed_dim=num_hid, 
                num_heads=num_head, 
                feed_forward_dim=num_feed_forward, 
                rate=dropout,
                activation=tformer_activation,
                init=tformer_init,
                mha_dropout=tformer_mha_dropout,
                ffn_dropout=tformer_ffn_dropout,
                first_drop_path=tformer_first_drop_path,
                second_drop_path=tformer_second_drop_path
            ) for _ in range(num_layers)]
        )

    def forward(self, seq, seq_mask):
        # seq = (batch, seq_len, features)
        seq = self.seq_emb(seq)

        for l in self.encoder:
            seq = l(seq, seq_mask)

        return seq
    

class SeqEmbedding(nn.Module):

    def __init__(self, in_channels, num_hid, max_seq_len):
        super().__init__()
        linear1 = nn.Linear(in_features=in_channels, out_features=num_hid, bias=False)
        linear2 = nn.Linear(in_features=num_hid, out_features=num_hid, bias=False)

        self.emb = nn.Sequential(
            linear1,
            nn.GELU(),
            linear2
        )
        self.nan_emb = nn.parameter.Parameter(
            data=torch.zeros((num_hid), dtype=torch.float32),
            requires_grad=True
        )
        self.pos_emb = nn.parameter.Parameter(
            data=torch.zeros((max_seq_len, num_hid), dtype=torch.float32),
            requires_grad=True
        )

        fan = torch.nn.init._calculate_correct_fan(self.emb[2].weight, mode='fan_in')
        bound = np.sqrt(6 / fan)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.emb[0].weight)
            torch.nn.init.uniform_(self.emb[2].weight, a=-bound, b=bound)

    def forward(self, seq):
        seq = torch.where(
            # Checks whether landmark is missing in frame
            seq.sum(2, keepdims=True) == 0,
            # If so, the empty embedding is used
            self.nan_emb,
            # Otherwise the landmark data is embedded
            self.emb(seq),
        )
        seq = seq + self.pos_emb
        return seq

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_of_heads, dropout):
        super().__init__()
        depth = d_model // num_of_heads
        self.num_of_heads = num_of_heads
        self.wq = torch.nn.ModuleList([torch.nn.Linear(in_features=d_model, out_features=depth//2, bias=False) for _ in range(num_of_heads)])
        self.wk = torch.nn.ModuleList([torch.nn.Linear(in_features=d_model, out_features=depth//2, bias=False) for _ in range(num_of_heads)])
        self.wv = torch.nn.ModuleList([torch.nn.Linear(in_features=d_model, out_features=depth//2, bias=False) for _ in range(num_of_heads)])
        self.wo = torch.nn.Linear(in_features=(depth//2)*self.num_of_heads, out_features=d_model, bias=False)
        self.do = nn.Dropout(dropout)
        self.register_buffer(name='scale', tensor=torch.sqrt(torch.FloatTensor([depth])))

        with torch.no_grad():
            for i in range(num_of_heads):
                torch.nn.init.xavier_uniform_(self.wq[i].weight)
                torch.nn.init.xavier_uniform_(self.wk[i].weight)
                torch.nn.init.xavier_uniform_(self.wv[i].weight)
            torch.nn.init.xavier_uniform_(self.wo.weight)

    def scaled_dot_product(self, q, k, v, attention_mask):
        qkt = torch.matmul(q,k.permute(0,2,1))
        scaled_qkt = qkt/self.scale
        scaled_qkt = scaled_qkt.masked_fill(attention_mask, -1e10)
        scaled_qkt = F.softmax(scaled_qkt, dim=-1)
        z = torch.matmul(scaled_qkt, v)
        return z

    def forward(self, q, k, v, attention_mask=None):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](q)
            K = self.wk[i](k)
            V = self.wv[i](v)
            multi_attn.append(self.scaled_dot_product(Q,K,V,attention_mask))
        multi_head = torch.cat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        multi_head_attention = self.do(multi_head_attention)

        return multi_head_attention

class TransformerEncoder(nn.Module):

    def __init__(self,embed_dim, num_heads, feed_forward_dim, rate=0.1, activation='relu', init='he', mha_dropout=0.0, ffn_dropout=0.0, first_drop_path=0.0, second_drop_path=0.0):
        super().__init__()
        self.lnorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, num_heads, mha_dropout)
        self.lnorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

        linear1 = nn.Linear(in_features=embed_dim, out_features=feed_forward_dim, bias=False)
        linear2 = nn.Linear(in_features=feed_forward_dim, out_features=embed_dim, bias=False)

        self.ffn = nn.Sequential(
            linear1,
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            linear2,
        )

        fan = torch.nn.init._calculate_correct_fan(self.ffn[3].weight, mode='fan_in')
        bound = np.sqrt(6 / fan)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.ffn[0].weight)
            torch.nn.init.uniform_(self.ffn[3].weight, a=-bound, b=bound)

    def forward(self, seq, seq_mask):
        # seq = (batch, num_hid, len)
        # seq = (batch, len, num_hid)
        seq = self.lnorm1(seq + self.attn(seq,seq,seq,seq_mask))
        seq = self.lnorm2(seq + self.ffn(seq))

        return seq