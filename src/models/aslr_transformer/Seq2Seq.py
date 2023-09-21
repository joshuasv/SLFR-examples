NUM_CLASSES = 62
SEQ_CHANNELS = 88 # 44(kps)*2(xyz)

import torch
import pandas as pd
import torch.nn as nn

from src.models.aslr_transformer.Encoder import TransformerEncoder, SeqEmbedding
from src.models.aslr_transformer.Decoder import PhraseEmbedding, TransformerDecoder
from src.globals import SEQ_PAD_VALUE, PHRASE_PAD_VALUE, PHRASE_SOS_IDX

class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder, seq_pad_idx=SEQ_PAD_VALUE, phrase_pad_idx=PHRASE_PAD_VALUE):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_pad_idx = seq_pad_idx
        self.phrase_pad_idx = phrase_pad_idx

    def forward(self, seq, phrase):
        phrase_len = phrase.shape[1]
        seq_mask = (seq==SEQ_PAD_VALUE).all(2).to(seq.device)
        phrase_mask_key = (phrase==PHRASE_PAD_VALUE).to(phrase.device)
        phrase_mask_attn = torch.logical_not(torch.tril(torch.ones(phrase_len, phrase_len)).bool()).to(phrase.device)

        enc_seq = self.encoder(seq, seq_mask)
        out, _ = self.decoder(phrase, enc_seq, phrase_mask_key, phrase_mask_attn, seq_mask)
        
        return out, None


class Transformer(nn.Module):
    
    def __init__(
            self,
            num_hid=64,
            num_head=2,
            num_feed_forward=128,
            target_maxlen=100,
            num_layers_enc=4,
            num_layers_dec=1,
            num_classes=10,
            num_channels_seq_emb=132
        ):
        super().__init__()
        self.target_maxlen = target_maxlen
        self.seq_emb = SeqEmbedding(in_channels=num_channels_seq_emb, num_hid=num_hid)
        self.phrase_emb = PhraseEmbedding(vocab_len=num_classes, max_phrase_len=target_maxlen, hid_dim=num_hid)

        self.encoder = nn.ModuleList(
            [TransformerEncoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers_enc)]
        )

        self.decoder = nn.ModuleList(
            [TransformerDecoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers_dec)]
        )

        self.classifier = nn.Linear(in_features=num_hid, out_features=num_classes)

    def forward(self, seq, phrase):
        seq = seq.permute(0,2,1)
        # seq = (batch, channels, len)
        phrase_len = phrase.shape[1]
        seq_mask = (seq==SEQ_PAD_VALUE).all(1).to(seq.device)
        phrase_mask_key = (phrase==PHRASE_PAD_VALUE).to(phrase.device)
        phrase_mask_attn = torch.logical_not(torch.tril(torch.ones(phrase_len, phrase_len)).bool()).to(phrase.device)
        seq = self.seq_emb(seq)
        for l in self.encoder:
            seq = l(seq, seq_mask)
        trans = self.decode(seq, phrase, phrase_mask_key, phrase_mask_attn, seq_mask)
        out = self.classifier(trans)
        
        return out
    
    def decode(self, enc_out, phrase, seq_mask, phrase_mask_key, phrase_mask_attn):
        phrase = self.phrase_emb(phrase)
        for l in self.decoder:
            phrase = l(enc_out, phrase, seq_mask, phrase_mask_key, phrase_mask_attn)

        return phrase
    
    def generate(self, seq, sos_idx=PHRASE_SOS_IDX):
        """Perform inference over a batch of sequences using greedy decoding."""
        B, _, _ = seq.shape
        seq_mask = (seq==SEQ_PAD_VALUE).all(1).to(seq.device)
        enc_seq = self.seq_emb(seq)
        for l in self.encoder:
            enc_seq = l(enc_seq, seq_mask)
        trans = torch.empty((B,1), dtype=torch.int32).fill_(sos_idx).to(seq.device)
        trans_logits = []
        for i in range(self.target_maxlen - 1):
            phrase_mask_attn = torch.logical_not(torch.tril(torch.ones(i+1, i+1)).bool()).to(seq.device)

            dec_out = self.decode(enc_seq, trans, seq_mask, None, phrase_mask_attn)
            logits = self.classifier(dec_out)
            logits = logits.argmax(-1)
            last_logit = logits[:, -1][...,None]
            trans_logits.append(last_logit)
            trans = torch.concat((trans, last_logit), axis=-1)
        return trans
    
    def display_outputs(self, seq, phrase, idx_to_char, sos_idx=PHRASE_SOS_IDX):
        trans = self.generate(seq, sos_idx)
        trans = pd.DataFrame(trans.cpu().numpy()).applymap(lambda x: idx_to_char[x]).apply(lambda x: ''.join(x), axis=1).replace('<eos>.*', '<eos>', regex=True)
        phrase = pd.DataFrame(phrase.cpu().numpy()).applymap(lambda x: idx_to_char[x]).replace('<pad>', '').apply(lambda x: ''.join(x), axis=1)
        
        toshow = pd.concat([phrase, trans], axis=1)
        toshow.columns = ['phrase', 'pred']
        print(toshow.to_markdown(index=False))