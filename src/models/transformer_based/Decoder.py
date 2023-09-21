import torch
import torch.nn as nn

from src.models.transformer_based.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from src.models.transformer_based.PositionWiseFeedForwardLayer import PositionWiseFeedForwardLayer

from IPython import embed
from sys import exit

class Decoder(nn.Module):

    def __init__(
            self,
            output_dim, 
            hid_dim, 
            n_layers, 
            n_heads, 
            pf_dim, 
            dropout, 
            max_phrase_length=100
        ):
        super().__init__()
        self.max_phrase_length = max_phrase_length
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_phrase_length, hid_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(
                hid_dim=hid_dim, 
                n_heads=n_heads, 
                pf_dim=pf_dim, 
                dropout=dropout, 
            ) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(name='scale', tensor=torch.FloatTensor([hid_dim]))

    def forward(self, phrase, enc_seq, phrase_mask_key, phrase_mask_attn, seq_mask):
        # phrase = (batch_size, phrase_len)
        # enc_seq = (batch_size, seq_len, hid_dim)
        # phrase_mask = (batch_size, 1, phrase_len, phrase_len)
        # seq_mask = (batch_size, 1, 1, seq_len)
        batch_size = phrase.shape[0]
        phrase_len = phrase.shape[1]
        assert phrase_len <= self.max_phrase_length, f'decoder admits phrases of length <= {self.max_phrase_length}'

        pos = torch.arange(phrase_len).unsqueeze(0).repeat(batch_size, 1).to(phrase.device)
        # pos = (batch_size, phrase_len)
        phrase = self.dropout(
            (self.tok_embedding(phrase) * self.scale) + self.pos_embedding(pos)
        )
        # phrase = (batch_size, phrase_len, hid_dim)
        for layer in self.layers:
            phrase, attention = layer(phrase, enc_seq, phrase_mask_key, phrase_mask_attn, seq_mask)
        # phrase = (batch_size, phrase_len, hid_dim)
        # attention = (batch_size, n_heads, phrase_len, seq_len)

        output = self.fc_out(phrase)
        # output = (batch_size, phrase_len, output_dim)

        return output, attention


class DecoderLayer(nn.Module):

    def __init__(
            self,
            hid_dim, 
            n_heads, 
            pf_dim, 
            dropout, 
        ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        # self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        # self.enc_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.enc_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.poswise_ff = PositionWiseFeedForwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, phrase, enc_seq, phrase_mask_key, phrase_mask_attn, seq_mask):
        # phrase = (batch_size, phrase_len, hid_dim)
        # enc_seq = (batch_size, seq_len, hid_dim)
        # phrase_mask = (batch_size, 1, phrase_len, phrase_len)
        # seq_mask = (batch_size, 1, 1, seq_len)
        skip = phrase
        phrase, _ = self.self_attn(phrase, phrase, phrase, key_padding_mask=phrase_mask_key, attn_mask=phrase_mask_attn, need_weights=False)

        phrase = self.self_attn_layer_norm(skip + self.dropout(phrase))
        # phrase = (batch_size, phrase_len, hid_dim)

        # Encoder attention
        skip = phrase
        phrase, attention = self.enc_attn(phrase, enc_seq, enc_seq, key_padding_mask=seq_mask)
        # attention = (batch_size, n_heads, phrase_len, seq_len)

        phrase = self.enc_attn_layer_norm(skip + self.dropout(phrase))
        # phrase = (batch_size, phrase_len, hid_dim)

        skip = phrase
        phrase = self.poswise_ff(phrase)

        phrase = self.ff_layer_norm(skip + self.dropout(phrase))
        # phrase = (batch_size, phrase_len, hid_dim)

        return phrase, attention