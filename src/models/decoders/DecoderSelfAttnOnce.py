import torch
from torch import nn

from src.models.common.MultiHeadAttentionLayer import MultiHeadAttentionLayer


class Decoder(nn.Module):

    def __init__(self, num_hid, num_head, num_feed_forward, num_layers, num_classes, max_phrase_length, dropout=0.0) -> None:
        super().__init__()
        self.max_phrase_length = max_phrase_length
        self.phrase_emb = PhraseEmbedding(vocab_len=num_classes, max_phrase_len=max_phrase_length, hid_dim=num_hid)
        self.self_attn = MultiHeadAttentionLayer(hid_dim=num_hid, n_heads=num_head, dropout=dropout)
        self.lnorm = nn.LayerNorm(normalized_shape=num_hid)

        self.decoder = nn.ModuleList(
            [TransformerDecoder(num_hid, num_head, num_feed_forward, dropout) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(in_features=num_hid, out_features=num_classes)

    def forward(self, phrase, enc_out, phrase_mask, seq_mask):
        phrase = self.phrase_emb(phrase)

        skip = phrase
        phrase, _ = self.self_attn(phrase, phrase, phrase, mask=phrase_mask)
        phrase = phrase + skip
        phrase = self.lnorm(phrase)

        for l in self.decoder:
            phrase = l(enc_out, phrase, seq_mask, phrase_mask)


        out = self.classifier(phrase)

        return out
    
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

        # phrase = (batch, phrase_len, hid_dim)
        return phrase


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1) -> None:
        super().__init__()
        self.lnorm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.lnorm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.lnorm3 = nn.LayerNorm(normalized_shape=embed_dim)
        self.enc_attn = MultiHeadAttentionLayer(hid_dim=embed_dim, n_heads=num_heads, dropout=dropout_rate)

        self.dropout_high = nn.Dropout(0.5)
        self.dropout_low = nn.Dropout(dropout_rate)
        ffn = [
            nn.Linear(in_features=embed_dim, out_features=feed_forward_dim),
            nn.ReLU(),
            nn.Linear(in_features=feed_forward_dim, out_features=embed_dim),
        ]
        self.ffn = nn.Sequential(*ffn)

    # def forward(self, seq, phrase, seq_mask, phrase_mask_key, phrase_mask_attn):
    def forward(self, seq, phrase, seq_mask, phrase_mask):

        skip = phrase
        phrase, _ = self.enc_attn(phrase, seq, seq, mask=seq_mask)

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