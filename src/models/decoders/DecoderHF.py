import torch
from torch import nn

from src.tools import activation_factory, initialization_factory
from src.models.common.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from src.models.common.LateDropout import LateDropout
from src.models.common.DropPathLayer import DropPath
from src.models.common.FFN import FFN

from IPython import embed; from sys import exit

class Decoder(nn.Module):

    def __init__(
            self, 
            num_hid, 
            num_head, 
            num_feed_forward, 
            num_layers, 
            num_classes, 
            max_phrase_length, 
            learn_pos_enc=False,
            tformer_activation='relu',
            tformer_init='he',
            tformer_dropout=0.0, 
            tformer_attn_dropout=0.0,
            tformer_ffn_dropout=0.0,
            tformer_first_drop_path=0.0,
            tformer_second_drop_path=0.0,
            tformer_third_drop_path=0.0,
            late_dropout_rate=0.0,
            late_dropout_step=0,
            avg_pool=False,
        ) -> None:
        super().__init__()
        self.max_phrase_length = max_phrase_length
        self.avg_pool = avg_pool
        self.phrase_emb = PhraseEmbedding(
            vocab_len=num_classes, 
            max_phrase_len=max_phrase_length, 
            hid_dim=num_hid, 
            learn_pos_enc=learn_pos_enc
        )

        self.decoder = nn.ModuleList(
            [TransformerDecoder(
                embed_dim=num_hid, 
                num_heads=num_head, 
                feed_forward_dim=num_feed_forward, 
                activation=tformer_activation,
                init=tformer_init,
                dropout=tformer_dropout,
                attn_dropout=tformer_attn_dropout,
                ffn_dropout=tformer_ffn_dropout,
                first_drop_path=tformer_first_drop_path,
                second_drop_path=tformer_second_drop_path,
                third_drop_path=tformer_third_drop_path
            ) for _ in range(num_layers)]
        )

        if avg_pool:
            self.top_linear = nn.Linear(in_features=num_hid, out_features=num_hid*2)
            self.avg_pool = nn.AvgPool1d(kernel_size=num_hid*2)
        
        self.late_dropout = LateDropout(rate=late_dropout_rate, start_epoch=late_dropout_step)

        self.classifier = nn.Linear(in_features=1 if self.avg_pool else num_hid, out_features=num_classes)

    def forward(self, phrase, enc_out, phrase_mask, seq_mask):
        phrase = self.phrase_emb(phrase)

        for l in self.decoder:
            phrase = l(enc_out, phrase, seq_mask, phrase_mask)

        if self.avg_pool:
            phrase = self.top_linear(phrase)
            phrase = self.avg_pool(phrase)
        phrase = self.late_dropout(phrase)
        out = self.classifier(phrase)

        return out
    
class PhraseEmbedding(nn.Module):

    def __init__(self, vocab_len, max_phrase_len, hid_dim, learn_pos_enc=False):
        super().__init__()
        self.learn_pos_enc = learn_pos_enc
        self.phrase_emb = nn.Embedding(vocab_len, hid_dim)
        if learn_pos_enc:
            self.pos_emb = nn.parameter.Parameter(
                data=torch.zeros((max_phrase_len, hid_dim), dtype=torch.float32),
                requires_grad=True
            )
        else:
            self.pos_emb = nn.Embedding(max_phrase_len, hid_dim)

    def forward(self, phrase):
        # phrase = (batch, phrase_len)
        B, L = phrase.shape
        phrase = self.phrase_emb(phrase)
        if self.learn_pos_enc:
            pos = self.pos_emb[:L]
        else:
            pos = torch.arange(L).unsqueeze(0).repeat(B, 1).to(phrase.device)
            pos = self.pos_emb(pos)

        phrase = phrase + pos

        # phrase = (batch, phrase_len, hid_dim)
        return phrase


class TransformerDecoder(nn.Module):

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
            second_drop_path=0.0, 
            third_drop_path=0.0
        ) -> None:
        super().__init__()
        self.lnorm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.lnorm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.lnorm3 = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = MultiHeadAttentionLayer(hid_dim=embed_dim, n_heads=num_heads, dropout=attn_dropout)
        self.enc_attn = MultiHeadAttentionLayer(hid_dim=embed_dim, n_heads=num_heads, dropout=attn_dropout)
        self.ffn = FFN(
            input_dim=embed_dim, 
            expand_dim=feed_forward_dim, 
            activation=activation, 
            dropout=ffn_dropout
        )

        self.drop_path1 = DropPath(drop_prob=first_drop_path)
        self.drop_path2 = DropPath(drop_prob=second_drop_path)
        self.drop_path3 = DropPath(drop_prob=third_drop_path)

    def forward(self, seq, phrase, seq_mask, phrase_mask):

        # seq = (batch, hid_dim, len)
        # phrase = (batch, len, hid_dim)
        skip = phrase
        phrase, _ = self.self_attn(phrase, phrase, phrase, mask=phrase_mask)
        phrase = self.dropout(phrase)
        phrase = self.drop_path1(phrase) + skip
        phrase = self.lnorm1(phrase)

        skip = phrase
        phrase, _ = self.enc_attn(phrase, seq, seq, mask=seq_mask)
        # phrase = (batch, len, seq_hid_dim)
        phrase = self.dropout(phrase)
        phrase = self.drop_path2(phrase) + skip
        phrase = self.lnorm2(phrase)

        skip = phrase
        phrase = self.ffn(phrase)
        # phrase = (batch, len, hid_dim)
        phrase = self.dropout(phrase)
        phrase = self.drop_path3(phrase) + skip
        phrase = self.lnorm3(phrase)

        return phrase