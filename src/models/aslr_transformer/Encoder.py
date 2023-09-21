import torch.nn as nn
import torch.nn.functional as F

from IPython import embed; from sys import exit


class Encoder(nn.Module):

    def __init__(self, in_channels, num_hid, num_head, num_feed_forward, num_layers):
        super().__init__()
        self.seq_emb = SeqEmbedding(in_channels=in_channels, num_hid=num_hid)
        self.encoder = nn.ModuleList(
            [TransformerEncoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers)]
        )

    def forward(self, seq, seq_mask):
        # seq = (batch, seq_len, features)
        seq = seq.permute(0,2,1)
        seq = self.seq_emb(seq)
        seq = seq.permute(0,2,1)
        
        for l in self.encoder:
            seq = l(seq, seq_mask)

        return seq

class SeqEmbedding(nn.Module):

    def __init__(self, in_channels, num_hid) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=num_hid, kernel_size=11, padding='same')
        self.conv2 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, padding='same')
        self.conv3 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, padding='same')
    def forward(self, seq):
        # seq = (batch, channels, len)
        seq = F.relu(self.conv1(seq))
        seq = F.relu(self.conv2(seq))
        seq = F.relu(self.conv3(seq))
        # seq = (batch, num_hid, len)

        return seq
    

class TransformerEncoder(nn.Module):

    def __init__(self,embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        ffn = [
            nn.Linear(in_features=embed_dim, out_features=feed_forward_dim),
            nn.ReLU(),
            nn.Linear(in_features=feed_forward_dim, out_features=embed_dim),
        ]
        self.ffn = nn.Sequential(*ffn)
        self.lnorm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.lnorm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout = nn.Dropout(rate)

    def forward(self, seq, seq_mask):
        # seq = (batch, num_hid, len)
        # seq = (batch, len, num_hid)
        skip = seq
        seq, _ = self.attn(seq, seq, seq, key_padding_mask=seq_mask, need_weights=False)
        # seq = (batch, len, num_hid)
        seq = self.dropout(seq)
        seq = seq + skip
        seq = self.lnorm1(seq)
        skip = seq
        seq = self.ffn(seq)
        seq = self.dropout(seq)
        seq = self.lnorm2(seq)
        seq = seq + skip
        # seq = (batch, num_hid, len)

        return seq