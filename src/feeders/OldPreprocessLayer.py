from sys import exit 

import torch
from IPython import embed
from torch.nn import Module


def force_right_hand(x, dom_hand_idxs):
    n_frames = x.shape[0]
    xs = x[:n_frames,:543,0]
    if dom_hand_idxs[0] == 468:
        xs = 1 - xs
    xs = xs.unsqueeze(2)
    
    return xs

def update_wrist(x, n_frames, pose_idxs, dom_hand_idxs):
    pose = x[:n_frames, pose_idxs, 2]
    hand = x[:n_frames, dom_hand_idxs]
    if dom_hand_idxs[0] == 468:
        zwrist = pose[:n_frames,2].unsqueeze(1)
    else:
        zwrist = pose[:n_frames,3].unsqueeze(1)
    zs = hand[:n_frames,:21,2] + zwrist
    zs = zs.unsqueeze(2)

    return zs

def get_frames(x):
    n_frames = torch.tensor(x.shape[0])
    return n_frames

def drop_frames_from_dominant_hand(x, dom_hand_idxs):
    n_frames = x.shape[0]
    hand = x[:n_frames, dom_hand_idxs, :2]
    idxs_bool = torch.isnan(hand).all(2).all(1)
    x = x[~idxs_bool]

    return x

def pad(x, to_compute:bool):
    if to_compute:
        x = pad_repeat(x)
    else:
        x = pad_zeros(x)
    return x

def pad_zeros(x):
    if x.shape[0] >= 64:
        x = x[:64]
    else:
        to_pad = 64 - x.shape[0]
        zero_pad = torch.zeros((to_pad, x.shape[1], x.shape[2]))
        x = torch.cat((x, zero_pad), dim=0)
        
    return x

def pad_repeat(x):
    curr_n_frames = x.shape[0]
    if curr_n_frames >= 64:
        x = x[:64]
    else:
        to_pad = 64 - curr_n_frames
        repeat_pad = torch.cat([x[i % curr_n_frames].unsqueeze(0) for i in range(to_pad)], dim=0)
        x = torch.cat((x,repeat_pad), dim=0)
    return x


def select_dominant_hand(x, lhand_idxs, rhand_idxs):
    lhand = x[:, lhand_idxs, :2]
    rhand = x[:, rhand_idxs, :2]

    pct_lhand_missing = lhand[lhand!=lhand].numel() / lhand.numel()
    pct_rhand_missing = rhand[rhand!=rhand].numel() / rhand.numel()

    if pct_lhand_missing == 1.0:
        return rhand_idxs
    elif pct_rhand_missing == 1.0:
        return lhand_idxs
    else:
        if pct_lhand_missing > .9:
            return rhand_idxs
        elif pct_rhand_missing > .9:
            return lhand_idxs
    # If cannot be discarded by pct of missing hand
    # Compute energy
    lhandx = lhand[...,0]
    lhandy = lhand[...,1]
    rhandx = rhand[...,0]
    rhandy = rhand[...,1]
    lhand_energy = lhandx[~(lhandx!=lhandx)].std() + lhandy[~(lhandy!=lhandy)].std() 
    rhand_energy = rhandx[~(rhandx!=rhandx)].std() + rhandy[~(rhandy!=rhandy)].std() 

    if rhand_energy >= lhand_energy:
        return rhand_idxs
    else:
        return lhand_idxs
    
def compute_bones(x, bones, to_compute:bool):
    if to_compute:
        bone_data = x[:, bones[:, 0]] - x[:, bones[:, 1]]
        x = torch.cat((x, bone_data), dim=2)

    return x

def compute_angles(x, angles, to_compute:bool):
    # Adds one channel dimension
    if to_compute:
        the_joint = x[:, angles[:, 0], :3]
        v1 = x[:, angles[:, 1], :3]
        v2 = x[:, angles[:, 2], :3]
        vec1 = v1 - the_joint
        vec2 = v2 - the_joint
        angle_data = 1 - torch.nn.functional.cosine_similarity(vec1, vec2, 2, 0.0)
        angle_data = angle_data.unsqueeze(2)
        angle_data = torch.where(torch.isnan(angle_data), torch.tensor(0.0), angle_data)
        x = torch.cat((x,angle_data), dim=2)
    return x
    
def compute_fict_p(pose, to_compute:bool):
    if to_compute:
        mid_p = (pose[:, 0] + pose[:, 1]) / 2 # between both shoulders
        mid_p = mid_p.unsqueeze(1)
        pose = torch.cat((pose, mid_p),dim=1)
    
    return pose

def compute_vels(x, to_compute:bool):
    if to_compute:
        vel_data = x[1:] - x[:-1]
        pad_zeros = torch.zeros((1, vel_data.shape[1], vel_data.shape[2]), dtype=torch.float32)
        vel_data = torch.cat((vel_data, pad_zeros), dim=0)
        x = torch.cat((x, vel_data), dim=2)
    return x

def normalize(x, to_compute:bool):
    if to_compute:
        ref_p = x[:, 23]
        ref_p = ref_p.unsqueeze(1)
        x = x - ref_p

        # Get distance from left and right shoulder keypoints
        lshoulder = x[:, 19]
        rshoulder = x[:, 20]
        distance = (lshoulder - rshoulder)
        distance = distance * distance
        distance = torch.sum(distance, dim=1)
        distance = torch.sqrt(distance)
        distance = distance.unsqueeze(1).unsqueeze(2)
        x = torch.div(x, distance)

    return x

class PreprocessLayer(Module):
     
    def __init__(
        self, 
        prep_pad_repeat=False, 
        prep_compute_bones=False, 
        prep_compute_fict_p=False, 
        prep_normalize=False, 
        prep_compute_angles=False, 
        prep_compute_vels=False,
    ):
        super().__init__()
        self.prep_pad_repeat = prep_pad_repeat
        self.prep_compute_bones = prep_compute_bones
        self.prep_compute_fict_p = prep_compute_fict_p
        self.prep_normalize = prep_normalize
        self.prep_compute_angles = prep_compute_angles
        self.prep_compute_vels = prep_compute_vels
        self.idxs = [
            torch.tensor([468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488]),  # lhand
            torch.tensor([522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]),  # rhand
            torch.tensor([0, 1, 10, 17, 39, 61, 70, 105, 107, 137, 152, 181, 269, 291, 300, 323, 334, 336, 405]),                     # face
            torch.tensor([500, 501, 502, 503, 504, 505]),                                                                             # pose
            torch.tensor([500, 501, 502, 503]),                                                                                       # pose no wrists
        ]
        if prep_compute_fict_p:
            self.bones = torch.tensor([
                (0,12),(1,5),(2,8),(3,10),(4,0),(5,11),(6,9),(7,6),(8,7),(9,1),
                (10,23),(11,0),(12,13),(13,18),(14,15),(15,10),(16,17),(17,2),
                (18,3),(19,23),(20,23),(21,19),(22,20),(23,24),(24,22),(25,24),
                (26,25),(27,26),(28,27),(29,24),(30,29),(31,30),(32,31),(33,24),
                (34,33),(35,34),(36,35),(37,24),(38,37),(39,38),(40,39),(41,24),
                (42,41),(43,42),(44,43)
            ])
            self.angles = torch.tensor([
                (0,4,12),(1,9,5),(2,8,17),(3,11,18),(4,5,0),(5,4,11),(6,9,7),
                (7,6,8),(8,7,2),(9,6,1),(10,3,15),(11,5,3),(12,0,13),(13,12,18),
                (14,16,15),(15,14,10),(16,17,14),(17,2,16),(18,13,3),(19,21,23),
                (20,22,23),(21,19,22),(22,20,24),(23,10,19),(24,22,23),
                (25,24,26),(26,27,25),(27,28,26),(28,27,32),(29,24,30),
                (30,31,29),(31,32,30),(32,28,36),(33,24,34),(34,33,35),
                (35,34,36),(36,32,40),(37,24,38),(38,37,39),(39,38,40),
                (40,36,44),(41,24,42),(42,41,43),(43,42,44),(44,40,24)
            ])
        else:
            self.bones = torch.tensor([
                (0,12),(1,5),(2,8),(3,10),(4,0),(5,11),(6,9),(7,6),(8,7),(9,1),
                (10,23),(11,0),(12,13),(13,18),(14,15),(15,10),(16,17),(17,2),
                (18,3),(19,10),(20,10),(21,19),(22,20),(23,22),(24,23),(25,24),
                (26,25),(27,26),(28,23),(29,28),(30,29),(31,30),(32,23),(33,32),
                (34,33),(35,34),(36,23),(37,36),(38,37),(39,38),(40,23),(41,40),
                (42,41),(43,42)
            ])
            self.angles = torch.tensor([
                (0,4,12),(1,9,5),(2,8,17),(3,11,18),(4,5,0),(5,4,11),(6,9,7),
                (7,6,8),(8,7,2),(9,6,1),(10,3,15),(11,5,3),(12,0,13),(13,12,18),
                (14,16,15),(15,14,10),(16,17,14),(17,2,16),(18,13,3),(19,21,10),
                (20,22,10),(21,19,22),(22,20,23),(23,10,22),(24,25,23),
                (25,24,26),(26,27,25),(27,23,31),(28,23,29),(29,28,30),
                (30,31,29),(31,27,35),(32,23,33),(33,34,32),(34,33,35),
                (35,31,39),(36,23,37),(37,36,38),(38,37,39),(39,35,43),
                (40,23,41),(41,40,42),(42,41,43),(43,23,39)
            ])

        self.force_right_hand_fn = torch.jit.script(force_right_hand, example_inputs=[
            (torch.rand(12, 543, 3), self.idxs[0]), (torch.rand(12, 543, 3), self.idxs[1]),
        ])
        self.update_wrist_fn = torch.jit.script(update_wrist, example_inputs=[
            (torch.rand(12, 543, 3), 12, self.idxs[3], self.idxs[0]),
            (torch.rand(12, 543, 3), 12, self.idxs[3], self.idxs[1]),
        ])
        self.get_frames_fn = torch.jit.script(get_frames, example_inputs=[
            (torch.rand(123, 543, 3)), (torch.rand(1, 543, 3)),
        ])
        w_nans = torch.rand(123, 543, 3)
        w_nans[0:10, self.idxs[0], :2] = torch.nan
        self.drop_frames_from_dominant_hand_fn = torch.jit.script(drop_frames_from_dominant_hand, example_inputs=[
            (w_nans, self.idxs[0]), (torch.rand(44, 543, 3), self.idxs[1])
        ])
        self.pad_fn = torch.jit.script(pad, example_inputs=[
            (torch.rand(12, 543, 3), True), (torch.rand(121, 543, 3), False), (torch.rand(64, 543, 3), True)
        ])
        self.select_dominant_hand_fn = torch.jit.script(select_dominant_hand, example_inputs=[
            (w_nans, self.idxs[0], self.idxs[1]), (torch.rand(32, 543, 3), self.idxs[0], self.idxs[1]), 
        ])
        self.bones_fn = torch.jit.script(compute_bones, example_inputs=[
            (torch.rand(12, 44, 3), self.bones, True), (torch.rand(42, 44, 3), self.bones, False),
        ])
        self.angles_fn = torch.jit.script(compute_angles, example_inputs=[
            (torch.rand(12, 44, 3), self.angles, True), (torch.rand(42, 44, 3), self.angles, False),
        ])
        self.compute_fict_p_fn = torch.jit.script(compute_fict_p, example_inputs=[
            (torch.rand(12, 4, 3), True), (torch.rand(12, 4, 3), False)
        ])
        self.normalize_fn = torch.jit.script(normalize, example_inputs=[
            (torch.rand(12, 45, 3), True), (torch.rand(112, 45, 3), True), (torch.rand(33, 45, 3), False)
        ])
        self.vels_fn = torch.jit.script(compute_vels, example_inputs=[
            (torch.rand(12, 44, 3), True), (torch.rand(12, 44, 3), False)
        ])

    def forward(self, x):
        # Data comes with batch dimenion (frames, kps, xyz)
        dom_hand_idxs = self.select_dominant_hand_fn(x, self.idxs[0], self.idxs[1])
        x = self.drop_frames_from_dominant_hand_fn(x, dom_hand_idxs)

        x = torch.where(torch.isnan(x), torch.tensor(0.0), x)

        new_xs = self.force_right_hand_fn(x, dom_hand_idxs)
        n_frames = self.get_frames_fn(x)
        x = torch.cat([new_xs, x[:n_frames,:543,1:]], dim=2)

        face = x[:n_frames, self.idxs[2], :3]
        pose = x[:n_frames, self.idxs[4], :3]
        hand = x[:n_frames, dom_hand_idxs, :3]

        new_zs = self.update_wrist_fn(x, n_frames, self.idxs[3], dom_hand_idxs)
        hand = torch.cat((hand[:n_frames,:21,:2],new_zs), dim=2)

        pose = self.compute_fict_p_fn(pose, self.prep_compute_fict_p)

        x = torch.cat((face,pose,hand), dim=1)
        x = self.normalize_fn(x, self.prep_normalize)
        
        # Extract bones, angles
        x = self.bones_fn(x, self.bones, self.prep_compute_bones)
        x = self.angles_fn(x, self.angles, self.prep_compute_angles)

        # Compute velocities of all previous computed features
        x = self.vels_fn(x, self.prep_compute_vels)

        x = self.pad_fn(x, self.prep_pad_repeat)
        x = torch.moveaxis(x, (0,1,2), (1,2,0)) 
        x = x.unsqueeze(3)
        x = x.unsqueeze(0)

        # (xyz, frames, kps, pers)
        return x

if __name__ == '__main__':
    from IPython import embed
    from sys import exit
    from src.feeders.GASLFRDataset import GASLFRDataset
    from src.globals import IN_MAX_SEQ_LEN, KPS_IDXS_TO_INCLUDE, MEDIAPIPE_CENTER_IDX

    d = GASLFRDataset(split_csv_fpath='./data_gen/baseline/train_split.csv')
    x, y = d[0]

    prep_layer = PreprocessLayer()
    prep_layer(x)
    embed(); exit()