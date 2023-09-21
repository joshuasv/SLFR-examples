import torch
import numpy as np
import torch.nn as nn

LIPS_LANDMARK_IDXS = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
])
RHAND_LANDMARK_IDXS = np.array([
    522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542
])
LHAND_LANDMARK_IDXS = np.array([
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488
])
IDXS_TO_INCLUDE = np.sort(np.concatenate([LIPS_LANDMARK_IDXS, RHAND_LANDMARK_IDXS, LHAND_LANDMARK_IDXS]))

MEANS = np.load('/home/gts/projects/jsoutelo/GASLRF-remote/notebooks/debug_mark/MEANS.npy')
STDS = np.load('/home/gts/projects/jsoutelo/GASLRF-remote/notebooks/debug_mark/STDS.npy')
MEANS = torch.tensor(MEANS)
STDS = torch.tensor(STDS)

class Preprocess(nn.Module):

    def __init__(
            self, 
            max_seq_len,
            include_z,
            include_vels,
        ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.include_z = include_z
        self.include_vels = include_vels
        self.idxs_to_include =IDXS_TO_INCLUDE
        self.mult = 3 if include_z else 2
        self.n_features = self.mult * len(self.idxs_to_include)

    def forward(self, x):
        x = x.reshape(-1, 543, 3)
        x = x[:, self.idxs_to_include]
        if not self.include_z:
            x = x[...,:2]
        x = x.reshape(-1, self.n_features)
        x = torch.where(x.isnan(), 0.0, x)

        x = x[None]
        hands = x[:, :, :84]
        hands = torch.abs(hands)
        mask = hands.sum(2)
        mask = mask != 0
        x = x[mask][None]

        N_FRAMES = len(x[0])
        if N_FRAMES < self.max_seq_len:
            x = torch.cat((
                x,
                torch.zeros((1,self.max_seq_len-N_FRAMES,self.n_features), dtype=torch.float32)
            ),axis=1)
        else:
            x = x[:, :self.max_seq_len]

        x = torch.where(x == 0.0, 0.0, (x-MEANS)/STDS)

        x = x.squeeze(0)

        return x

