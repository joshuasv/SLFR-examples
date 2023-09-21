from IPython import embed
from sys import exit

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.globals import KPS_IDXS_TO_INCLUDE, MEDIAPIPE_CENTER_IDX, SEQ_PAD_VALUE

def filter_nans_torch(x, kps_to_include):
    # merge list of list and get kps idxs without duplicates
    kps_to_include = list(set(sum(kps_to_include, [])))
    mask = x[:, kps_to_include].isnan().all(2).all(1).logical_not()
    x = x[mask]
    return x

def nan_std_torch(x, center, dim, keepdim):
    d = x - center
    std = torch.sqrt((d * d).nanmean(dim=dim, keepdim=keepdim))
    return std

def select_dominant_hand(x, kps_lhand, kps_rhand):
    lhand = x[:, kps_lhand, :2]
    rhand = x[:, kps_rhand, :2]
    pct_lhand_missing = lhand[lhand!=lhand].numel() / lhand.numel()
    pct_rhand_missing = rhand[rhand!=rhand].numel() / rhand.numel()
    if pct_lhand_missing == 1.0:
        return kps_rhand
    elif pct_rhand_missing == 1.0:
        return kps_lhand
    else:
        if pct_lhand_missing > .9:
            return kps_rhand
        elif pct_rhand_missing > .9:
            return kps_lhand
    # If cannot be discarded by pct of missing hand
    # Compute energy
    lhandx = lhand[...,0]
    lhandy = lhand[...,1]
    rhandx = rhand[...,0]
    rhandy = rhand[...,1]
    lhand_energy = lhandx[~(lhandx!=lhandx)].std() + lhandy[~(lhandy!=lhandy)].std() 
    rhand_energy = rhandx[~(rhandx!=rhandx)].std() + rhandy[~(rhandy!=rhandy)].std() 

    if rhand_energy >= lhand_energy:
        return kps_rhand
    else:
        return kps_lhand

class Preprocess(nn.Module):

    def __init__(
            self, 
            max_seq_len,
            include_z,
            include_vels,
            kps_to_include=KPS_IDXS_TO_INCLUDE,
            central_point=MEDIAPIPE_CENTER_IDX,
            seq_pad_value=SEQ_PAD_VALUE
        ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.kps_to_include = kps_to_include
        self.include_z = include_z
        self.central_point = central_point
        self.include_vels = include_vels
        self.seq_pad_value = seq_pad_value

        self.mult = 3 if include_z else 2

    def forward(self, x):
        kps_dom_hand = select_dominant_hand(
            x, 
            kps_lhand=self.kps_to_include[0],
            kps_rhand=self.kps_to_include[1]
        )
        x = filter_nans_torch(x, self.kps_to_include)

        x = x[None,...] if x.ndim == 3 else x

        mean = x[:, :, self.central_point].nanmean(dim=1, keepdim=True)
        mean = torch.where(mean.isnan(), 0.5, mean)

        kps_to_include = self.kps_to_include[2] + self.kps_to_include[4] + kps_dom_hand
        x = x[:, :, kps_to_include]

        std = nan_std_torch(x, mean, dim=(1,2), keepdim=True)

        x = (x - mean) / std

        if self.max_seq_len is not None:
            x = x[:, :self.max_seq_len]
        seq_len = x.shape[1]

        if not self.include_z:
            x = x[..., :2]

        x = x.reshape(-1, seq_len, self.mult*len(kps_to_include))

        if self.include_vels:
            dx_p4d = (0, 0, 0, 0, 0, 1, 0, 0)
            dx2_p4d = (0, 0, 0, 0, 0, 2, 0, 0)
            dx = F.pad(x[:, 1:] - x[:, :-1], dx_p4d, 'constant', 0.0) if seq_len > 1 else torch.zeros_like(x)
            dx2 = F.pad(x[:, 2:] - x[:, :-2], dx2_p4d, 'constant', 0.0) if seq_len > 1 else torch.zeros_like(x)
            x = torch.cat((
                x,
                dx.reshape(-1, seq_len, self.mult*len(kps_to_include)),
                dx2.reshape(-1, seq_len, self.mult*len(kps_to_include))
            ), dim=-1)

        x = torch.where(x.isnan(), 0.0, x)

        if seq_len < self.max_seq_len:
            p3d = (0, 0, 0, self.max_seq_len - seq_len, 0, 0)
            x = F.pad(x, p3d, 'constant', self.seq_pad_value)

        return x