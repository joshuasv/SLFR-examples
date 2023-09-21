SEED = 42

EXP_MAX_TRAIN_DATA = 15000
TRAIN_CSV_FPATH = './data/etrain.csv'
SUPPLEMENTAL_CSV_FPATH = './data/esupplemental_metadata.csv'
TRAIN_DATA_DPATH = './data/train_landmarks_extracted'
SUPPLEMENTAL_DATA_DPATH = './data/supplemental_landmarks_extracted'
CHAR_TO_IDX_FPATH = './data/character_to_prediction_index.json'
NUM_CHARS = 59
TRAIN_MAX_PHRASE_LEN = 31
SUPP_MAX_PHRASE_LEN = 43

MEDIAPIPE_NUM_KEYPOINTS = 543
MEDIAPIPE_FACE_IDXS=list(range(0, 468))
MEDIAPIPE_LHAND_IDXS=list(range(468, 489))
MEDIAPIPE_POSE_IDXS=list(range(489, 522))
MEDIAPIPE_RHAND_IDXS=list(range(522, 543))
MEDIAPIPE_CENTER_IDX = 17 # Corresponds to the lowest keypoint in the lips
KPS_IDXS_TO_INCLUDE = [
    [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488],  # lhand
    [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542],  # rhand
    [0, 1, 10, 17, 39, 61, 70, 105, 107, 137, 152, 181, 269, 291, 300, 323, 334, 336, 405],                     # face
    [500, 501, 502, 503, 504, 505],                                                                             # pose
    [500, 501, 502, 503],                                                                                       # pose no wrists
]

# KPS_IDXS_TO_INCLUDE = [
#     [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488],  # lhand
#     [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542],  # rhand
#     [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
# ]

MAX_SEQ_LEN = 384
MAX_PHRASE_LEN = 31 + 1 + 1 # 31 + sos + eos
SEQ_PAD_VALUE = 61
PHRASE_SOS_IDX = 59
PHRASE_EOS_IDX = 60
PHRASE_PAD_VALUE = 61