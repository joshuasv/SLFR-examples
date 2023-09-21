import torch
import torch.nn as nn

from src.models.rwkv.RWKV_TimeMix import RWKV_TimeMix
from src.models.rwkv.RWKV_ChannelMix import RWKV_ChannelMix

class Encoder(nn.Module):

    def __init__(
            self,
            input_dim,
            hid_dim,
            n_layers,
            n_heads,
            ctx_len,
            n_ffn,
            vocab_size=62
        ):
        super().__init__()
        self.ctx_len = ctx_len

        self.seq_embedding = nn.Linear(input_dim, hid_dim)
        self.layers = nn.ModuleList(
            [
                RWKV(
                    hid_dim=hid_dim,
                    n_attn=hid_dim,
                    n_head=n_heads,
                    ctx_len=ctx_len,
                    n_ffn=n_ffn,
                )
            for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(hid_dim)
        self.time_out = nn.Parameter(torch.ones(1,ctx_len,1)) # reduce confidence of early tokens
        self.head = nn.Linear(hid_dim, vocab_size, bias=False)

        self.head_q = nn.Linear(hid_dim, 256)
        self.head_q.scale_init = 0.01
        self.head_k = nn.Linear(hid_dim, 256)
        self.head_k.scale_init = 0.01
        self.register_buffer("copy_mask", torch.tril(torch.ones(ctx_len, ctx_len)))

    def forward(self, x):
        # x = (batch_size, seq_len, n_features)
        B, T, C = x.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.seq_embedding(x)
        # x = (batch_size, seq_len, hid_dim)

        for l in self.layers:
            x = l(x)
        # x = (batch_size, seq_len, hid_dim)

        return x



class RWKV(nn.Module):
    
    def __init__(
            self,
            hid_dim,
            n_attn,
            n_head,
            ctx_len,
            n_ffn
        ):
        super().__init__()

        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)

        self.attn = RWKV_TimeMix(
            hid_dim=hid_dim,
            n_attn=n_attn,
            n_head=n_head,
            ctx_len=ctx_len,
        )
        self.mlp = RWKV_ChannelMix(hid_dim=hid_dim, n_ffn=n_ffn)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        
        return x
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.feeders.GASLFRDataset import GASLFRDataset
    from IPython import embed; from sys import exit

    m = Encoder(
        input_dim=88,
        hid_dim=256,
        n_layers=2,
        n_heads=4,
        ctx_len=384,
        n_ffn=256,
    )
    d = GASLFRDataset(
        split_csv_fpath='./data_gen/baseline/train_split.csv',
        max_phrase_len=45,
        prep_max_seq_len=384,
        prep_include_z=False,
        prep_include_vels=False,
        debug=False,
    )
    dl = DataLoader(
        dataset=d,
        batch_size=12,
        num_workers=0
    )
    X, y = next(iter(dl))

    o = m(X)
    embed(); exit()