import torch
import torch.nn as nn
import torch.nn.functional as F

class RWKV_ChannelMix(nn.Module):
    def __init__(
            self,
            hid_dim,
            n_ffn,
        ):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        
        hidden_sz = 5 * n_ffn // 2 # can use smaller hidden_sz because of receptance gating
        self.key = nn.Linear(hid_dim, hidden_sz)
        self.value = nn.Linear(hid_dim, hidden_sz)
        self.weight = nn.Linear(hidden_sz, hid_dim)
        self.receptance = nn.Linear(hid_dim, hid_dim)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        
        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        
        wkv = self.weight(F.mish(k) * v) # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv