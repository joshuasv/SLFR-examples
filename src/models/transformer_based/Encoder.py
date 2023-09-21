import torch
import torch.nn as nn

from src.models.transformer_based.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from src.models.transformer_based.PositionWiseFeedForwardLayer import PositionWiseFeedForwardLayer


class Encoder(nn.Module):

    def __init__(
            self,
            input_dim, 
            hid_dim, 
            n_layers, 
            n_heads, 
            pf_dim,
            dropout, 
            max_seq_length=100
        ):
        super().__init__()

        self.seq_embedding = nn.Linear(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, hid_dim)
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

        batch_size = seq.shape[0]
        seq_len = seq.shape[1]

        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(seq.device)
        # pos = (batch_size, seq_len)

        seq = self.dropout(
            (self.seq_embedding(seq) * self.scale) + self.pos_embedding(pos)
        )

        # seq = seq.permute(0,2,1)
        # seq = self.seq_emb(seq)
        # seq = seq.permute(0,2,1)
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
        ):
        super().__init__()

        # self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=n_heads, batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.poswise_ff = PositionWiseFeedForwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, seq_mask):
        # seq = (batch_size, seq_len, seq_features)
        # seq_mask = (batch_size, 1, 1, seq_len)

        seq_residual, _ = self.self_attn(seq, seq, seq, key_padding_mask=seq_mask, need_weights=False)

        seq = self.self_attn_layer_norm(seq + self.dropout(seq_residual))
        # seq = (batch_size, seq_len, hid_dim)

        seq_residual = self.poswise_ff(seq)

        seq = self.ff_layer_norm(seq + self.dropout(seq_residual))
        # seq = (batch_size, seq_len, hid_dim)

        return seq