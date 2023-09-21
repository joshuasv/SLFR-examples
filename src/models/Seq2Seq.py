import torch
import pandas as pd
import torch.nn as nn

from src.globals import SEQ_PAD_VALUE, PHRASE_PAD_VALUE, PHRASE_SOS_IDX

class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder, seq_pad_idx=SEQ_PAD_VALUE, phrase_pad_idx=PHRASE_PAD_VALUE):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_pad_idx = seq_pad_idx
        self.phrase_pad_idx = phrase_pad_idx

    def forward(self, seq, phrase):
        seq_mask = self.make_seq_mask(seq).to(seq.device)
        phrase_mask = self.make_phrase_mask(phrase).to(seq.device)
        # phrase_mask_key, phrase_mask_attn = phrase_masks[0].to(phrase.device), phrase_masks[1].to(phrase.device)
        enc_seq = self.encoder(seq, seq_mask)
        out = self.decoder(phrase, enc_seq, phrase_mask, seq_mask)
        
        return out, None

    def make_seq_mask(self, seq):
        # seq = (batch_size, seq_len, seq_features)
        seq_mask = (seq != self.seq_pad_idx).all(2).unsqueeze(1).unsqueeze(2)
        # seq_mask = (batch_size, 1, 1, seq_len)

        return seq_mask

    def make_phrase_mask(self, phrase):
        # phrase = (batch_size, phrase_len)
        phrase_pad_mask = (phrase != self.phrase_pad_idx).unsqueeze(1).unsqueeze(2)
        # phrase_pad_mask = (batch_size, 1, 1, phrase_len)

        phrase_len = phrase.shape[1]
        phrase_sub_mask = torch.tril(torch.ones(phrase_len, phrase_len)).bool()
        phrase_sub_mask = phrase_sub_mask.to(phrase.device)
        # phrase_sub_mask = (phrase_len, phrase_len)

        phrase_mask = phrase_pad_mask & phrase_sub_mask
        # phrase_mask = (batch_size, 1, phrase_len, phrase_len)

        return phrase_mask
    
    def greedy_translate_sequence(self, seq, sos_idx=PHRASE_SOS_IDX):
        """Perform inference over a batch of sequences using greedy decoding."""
        self.eval()
        B, _, _ = seq.shape
        seq_mask = self.make_seq_mask(seq)
        enc_seq = self.encoder(seq, seq_mask)
        trans = torch.empty((B,1), dtype=torch.int32).fill_(sos_idx).to(seq.device)
        trans_logits = []
        max_phrase_len = self.decoder.max_phrase_length
        for i in range(max_phrase_len - 1):
            phrase_mask = self.make_phrase_mask(trans)
            dec_out = self.decoder(trans, enc_seq, phrase_mask, seq_mask)
            logits = dec_out.argmax(-1)
            last_logit = logits[:, -1][...,None]
            trans_logits.append(last_logit)
            trans = torch.concat((trans, last_logit), axis=-1)

        return trans