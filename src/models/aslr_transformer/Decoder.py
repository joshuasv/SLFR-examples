import math

import torch
from torch import nn, Tensor

class Decoder(nn.Module):

    def __init__(self, num_hid, num_head, num_feed_forward, num_layers, num_classes, max_phrase_length) -> None:
        super().__init__()
        self.max_phrase_length = max_phrase_length
        self.phrase_emb = PhraseEmbedding(vocab_len=num_classes, max_phrase_len=max_phrase_length, hid_dim=num_hid)

        self.decoder = nn.ModuleList(
            [TransformerDecoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(in_features=num_hid, out_features=num_classes)

    def forward(self, phrase, enc_out, phrase_mask_key, phrase_mask_attn, seq_mask):
        phrase = self.phrase_emb(phrase)
        for l in self.decoder:
            phrase = l(enc_out, phrase, seq_mask, phrase_mask_key, phrase_mask_attn)


        out = self.classifier(phrase)

        return out, None
    
class PhraseEmbedding(nn.Module):

    def __init__(self, vocab_len, max_phrase_len, hid_dim):
        super().__init__()
        self.phrase_emb = nn.Embedding(vocab_len, hid_dim)
        self.pos_emb = nn.Embedding(max_phrase_len, hid_dim)

    def forward(self, phrase):
        # phrase = (batch, phrase_len)
        B, L = phrase.shape
        pos = torch.arange(L).unsqueeze(0).repeat(B, 1).to(phrase.device)
        phrase = self.phrase_emb(phrase)
        pos = self.pos_emb(pos)
        phrase = phrase + pos

        # phrase = (batch, hid_dim)
        return phrase


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1) -> None:
        super().__init__()
        self.lnorm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.lnorm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.lnorm3 = nn.LayerNorm(normalized_shape=embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.enc_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.dropout_high = nn.Dropout(0.5)
        self.dropout_low = nn.Dropout(dropout_rate)
        ffn = [
            nn.Linear(in_features=embed_dim, out_features=feed_forward_dim),
            nn.ReLU(),
            nn.Linear(in_features=feed_forward_dim, out_features=embed_dim),
        ]
        self.ffn = nn.Sequential(*ffn)

    def forward(self, seq, phrase, seq_mask, phrase_mask_key, phrase_mask_attn):
        # seq = (batch, hid_dim, len)
        # phrase = (batch, len, hid_dim)
        skip = phrase
        phrase, _ = self.self_attn(phrase, phrase, phrase, key_padding_mask=phrase_mask_key, attn_mask=phrase_mask_attn, need_weights=False)
        phrase = self.dropout_high(phrase)
        phrase = phrase + skip
        phrase = self.lnorm1(phrase)

        skip = phrase
        phrase, _ = self.enc_attn(phrase, seq, seq, key_padding_mask=seq_mask, need_weights=False)
        # phrase = (batch, len, seq_hid_dim)
        phrase = self.dropout_low(phrase)
        phrase = phrase + skip
        phrase = self.lnorm2(phrase)

        skip = phrase
        phrase = self.ffn(phrase)
        # phrase = (batch, len, hid_dim)
        phrase = self.dropout_low(phrase)
        phrase = phrase + skip
        phrase = self.lnorm3(phrase)

        return phrase



# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         # pe = torch.zeros(max_len, 1, d_model)
#         pe = torch.zeros(1, max_len, d_model)
#         # pe[:, 0, 0::2] = torch.sin(position * div_term)
#         # pe[:, 0, 1::2] = torch.cos(position * div_term)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Arguments:
#             # x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#             x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
#         """
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)