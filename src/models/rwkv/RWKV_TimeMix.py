import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKV_TimeMix(nn.Module):
    def __init__(
            self,
            hid_dim,
            n_attn,
            n_head,
            ctx_len
        ):
        super().__init__()
        assert n_attn % n_head == 0
        self.ctx_len = ctx_len
        self.n_head = n_head
        self.head_size = n_attn // n_head

        with torch.no_grad(): # initial time_w curves for better convergence
            ww = torch.ones(n_head, ctx_len)
            curve = torch.tensor([-(ctx_len - 1 - i) for i in range(ctx_len)]) # the distance
            for h in range(n_head):
                if h < n_head - 1:
                    decay_speed = math.pow(ctx_len, -(h+1)/(n_head-1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(ctx_len, 1))
                
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.key = nn.Linear(hid_dim, n_attn)
        self.value = nn.Linear(hid_dim, n_attn)
        self.receptance = nn.Linear(hid_dim, n_attn)

        # if rwkv_tiny_attn > 0:
        #     self.tiny_att = RWKV_TinyAttn(config)

        self.output = nn.Linear(n_attn, hid_dim)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        # if hasattr(self, 'tiny_att'):
        #     tiny_att = self.tiny_att(x, self.mask)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60) # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        # if hasattr(self, 'tiny_att'):
        #     rwkv += tiny_att

        return rwkv * self.time_gamma[:T, :]