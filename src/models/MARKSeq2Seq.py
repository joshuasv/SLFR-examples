import torch
import pandas as pd
import torch.nn as nn

from src.globals import SEQ_PAD_VALUE, PHRASE_PAD_VALUE, PHRASE_SOS_IDX

from IPython import embed; from sys import exit

class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder, seq_pad_idx=SEQ_PAD_VALUE, phrase_pad_idx=PHRASE_PAD_VALUE, seq_mask_nan=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_pad_idx = seq_pad_idx
        self.phrase_pad_idx = phrase_pad_idx
        self.seq_mask_nan = seq_mask_nan

        is_enc_dec_dim_diff = encoder.num_hid != decoder.num_hid
        if is_enc_dec_dim_diff:
            self.proj = nn.Linear(in_features=encoder.num_hid, out_features=decoder.num_hid, bias=False)
            with torch.no_grad():
                torch.nn.init.xavier_uniform_(self.proj.weight)
        self.is_enc_dec_dim_diff = is_enc_dec_dim_diff

    def forward(self, seq, phrase):
        B, L = phrase.shape
        T = seq.shape[1]
        device = seq.device
        # True means not to attend
        phrase = nn.functional.pad(phrase, (0,T-L), value=PHRASE_PAD_VALUE)
        seq_mask = (seq==0.0).all(2).unsqueeze(1).to(device)
        phrase_mask = torch.logical_not(torch.tril(torch.ones(B,T,T)).to(torch.bool)).to(device)
        # enc_seq_phrase_mask = seq_mask.repeat(1,L,1)

        enc_seq = self.encoder(seq, seq_mask)
        if self.is_enc_dec_dim_diff:
            enc_seq = self.proj(enc_seq)
        out = self.decoder(phrase, enc_seq, phrase_mask, phrase_mask)
        
        return out, None

    def greedy_translate_sequence(self, seq, sos_idx=PHRASE_SOS_IDX):
        """Perform inference over a batch of sequences using greedy decoding."""
        self.eval()
        B, T, _ = seq.shape
        device = seq.device

        seq_mask = (seq==0.0).all(2).unsqueeze(1).to(device)
        seq_mask = (seq==self.seq_pad_idx).all(2).unsqueeze(1).to(device) 
        enc_seq = self.encoder(seq, seq_mask)
        if self.is_enc_dec_dim_diff:
            enc_seq = self.proj(enc_seq)

        trans = torch.empty((B,1), dtype=torch.int32).fill_(sos_idx).to(seq.device)
        trans_logits = []
        max_phrase_len = self.decoder.max_phrase_length
        for i in range(max_phrase_len - 1):
            L = i+1
            phrase_mask = torch.logical_not(torch.tril(torch.ones(B,L,1)).to(torch.bool)).to(device)
            dec_out = self.decoder(trans, enc_seq, phrase_mask, phrase_mask)
            
            logits = dec_out.argmax(-1)
            last_logit = logits[:, -1][...,None]
            trans_logits.append(last_logit)
            trans = torch.concat((trans, last_logit), axis=-1)

        return trans
