import torch
from torch import nn
import numpy as np

from src.tools import activation_factory, initialization_factory
from src.models.encoders.MARK import MultiHeadAttention
from src.models.common.LateDropout import LateDropout
from src.models.common.DropPathLayer import DropPath

from IPython import embed; from sys import exit; 

from src.globals import PHRASE_PAD_VALUE

class Decoder(nn.Module):

    def __init__(
            self, 
            num_hid, 
            num_head, 
            num_feed_forward, 
            num_layers, 
            num_classes, 
            max_phrase_length, 
            dropout=0.0, 
            learn_pos_enc=False,
            tformer_activation='relu',
            tformer_init='he',
            tformer_mha_dropout=0.0,
            tformer_ffn_dropout=0.0,
            tformer_first_drop_path=0.0,
            tformer_second_drop_path=0.0,
            late_dropout_rate=0.0,
            late_dropout_step=0,
        ) -> None:
        super().__init__()
        self.num_hid = num_hid
        self.max_phrase_length = max_phrase_length
        self.phrase_emb = PhraseEmbedding(
            vocab_len=num_classes, 
            max_phrase_len=max_phrase_length, 
            hid_dim=num_hid, 
            learn_pos_enc=learn_pos_enc
        )

        self.self_attn = MultiHeadAttention(num_hid, num_head, tformer_mha_dropout)
        self.lnorm = nn.LayerNorm(normalized_shape=num_hid)

        self.decoder = nn.ModuleList(
            [TransformerDecoder(
                embed_dim=num_hid, 
                num_heads=num_head, 
                feed_forward_dim=num_feed_forward,
                dropout_rate=dropout,
                activation=tformer_activation,
                init=tformer_init,
                mha_dropout=tformer_mha_dropout,
                ffn_dropout=tformer_ffn_dropout,
                first_drop_path=tformer_first_drop_path,
                second_drop_path=tformer_second_drop_path
            ) for _ in range(num_layers)]
        )
        self.late_dropout = nn.Dropout(late_dropout_rate)
        self.classifier = nn.Linear(in_features=num_hid, out_features=num_classes,bias=False)

        fan = torch.nn.init._calculate_correct_fan(self.classifier.weight, mode='fan_in')
        bound = np.sqrt(6 / fan)
        with torch.no_grad():
            torch.nn.init.uniform_(self.classifier.weight, a=-bound, b=bound)

    def forward(self, phrase, enc_out, phrase_mask, seq_mask):
        phrase = self.phrase_emb(phrase)

        phrase = self.lnorm(phrase + self.self_attn(phrase,phrase,phrase,phrase_mask))

        for l in self.decoder:
            phrase = l(enc_out, phrase, seq_mask)
        phrase = phrase[:, :self.max_phrase_length-1]

        out = self.classifier(phrase)

        return out
    
class PhraseEmbedding(nn.Module):

    def __init__(self, vocab_len, max_phrase_len, hid_dim, learn_pos_enc=False):
        super().__init__()
        self.learn_pos_enc = learn_pos_enc
        self.phrase_emb = nn.Embedding(vocab_len, hid_dim)
        self.pos_emb = nn.parameter.Parameter(
            data=torch.zeros((128, hid_dim), dtype=torch.float32),
            requires_grad=True
        )
        with torch.no_grad():
            torch.nn.init.zeros_(self.phrase_emb.weight)
            
    def forward(self, phrase):
        _, L = phrase.shape
        return self.phrase_emb(phrase) + self.pos_emb[:L]


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1, activation='relu', init='he', mha_dropout=0.0, ffn_dropout=0.0, first_drop_path=0.0, second_drop_path=0.0) -> None:
        super().__init__()
        activation_fn = activation_factory(activation)
        self.lnorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.enc_attn = MultiHeadAttention(embed_dim,num_heads,mha_dropout)
        self.lnorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        
        linear1 = nn.Linear(in_features=embed_dim, out_features=feed_forward_dim, bias=False)
        linear2 = nn.Linear(in_features=feed_forward_dim, out_features=embed_dim, bias=False)
        self.ffn = nn.Sequential(
            linear1,
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            linear2,
        )
        with torch.no_grad():
            fan = torch.nn.init._calculate_correct_fan(self.ffn[3].weight, mode='fan_in')
            bound = np.sqrt(6 / fan)
            with torch.no_grad():
                torch.nn.init.xavier_uniform_(self.ffn[0].weight)
                torch.nn.init.uniform_(self.ffn[3].weight, a=-bound, b=bound)


    def forward(self, seq, phrase, mask):
        phrase = self.lnorm1(phrase + self.enc_attn(phrase,seq,seq,mask))
        phrase = self.lnorm2(phrase + self.ffn(phrase))


        return phrase