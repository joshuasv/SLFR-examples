import torch
import torch.nn as nn

from src.globals import SEQ_PAD_VALUE, PHRASE_PAD_VALUE, PHRASE_SOS_IDX, PHRASE_EOS_IDX

from IPython import embed; from sys import exit

def make_seq_mask(seq, seq_pad_idx):
    seq_mask = (seq == seq_pad_idx).all(2)

    return seq_mask

def make_phrase_mask(phrase, phrase_pad_idx):
    phrase_len = phrase.shape[1]
    phrase_mask_key = (phrase==phrase_pad_idx)
    phrase_mask_attn = torch.logical_not(torch.tril(torch.ones(phrase_len, phrase_len)).bool())

    return phrase_mask_key, phrase_mask_attn


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, seq_pad_idx=SEQ_PAD_VALUE, phrase_pad_idx=PHRASE_PAD_VALUE):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_pad_idx = seq_pad_idx
        self.phrase_pad_idx = phrase_pad_idx

    def forward(self, seq, phrase):
        seq_mask = make_seq_mask(seq, self.seq_pad_idx).to(seq.device)
        phrase_mask_key, phrase_mask_attn = make_phrase_mask(phrase, self.phrase_pad_idx)
        phrase_mask_key = phrase_mask_key.to(phrase.device)
        phrase_mask_attn = phrase_mask_attn.to(phrase.device)
        # seq_mask = (batch_size, 1, 1, seq_len)
        # phrase_mask = (batch_size, 1, phrase_len, phrase_len)

        enc_seq = self.encoder(seq, seq_mask)
        # enc_seq = (batch_size, seq_len, hid_dim)

        output, attention = self.decoder(phrase, enc_seq, phrase_mask_key, phrase_mask_attn, seq_mask)
        # output = (batch_size, phrase_len, output_dim)
        # attention = (batch_size, n_heads, phrase_len, seq_len)

        return output, attention


def greedy_translate_sequence(seq, model, sos_idx=PHRASE_SOS_IDX, seq_pad_idx=SEQ_PAD_VALUE):
    """Perform inference over a batch of sequences using greedy decoding."""
    # seq = (batch)
    model.eval()
    B, _, _ = seq.shape
    seq_mask = make_seq_mask(seq, seq_pad_idx=seq_pad_idx)
    enc_seq = model.encoder(seq, seq_mask)
    trans = torch.empty((B,1), dtype=torch.int32).fill_(sos_idx).to(seq.device)
    trans_logits = []
    max_phrase_len = model.decoder.max_phrase_length
    for i in range(max_phrase_len - 1):
        phrase_mask_attn = torch.logical_not(torch.tril(torch.ones(i+1, i+1)).bool()).to(seq.device)
        dec_out, _ = model.decoder(trans, enc_seq, None, phrase_mask_attn, seq_mask)
        logits = dec_out.argmax(-1)
        last_logit = logits[:, -1][...,None]
        trans_logits.append(last_logit)
        trans = torch.concat((trans, last_logit), axis=-1)

    return trans


