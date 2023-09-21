from time import time

import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from src.globals import CHAR_TO_IDX_FPATH, MEDIAPIPE_NUM_KEYPOINTS, PHRASE_SOS_IDX, PHRASE_EOS_IDX, MAX_PHRASE_LEN, PHRASE_PAD_VALUE, EXP_MAX_TRAIN_DATA, SEED, KPS_IDXS_TO_INCLUDE
from utils import read_json
from src.feeders.Preprocess import Preprocess
from utils import import_class

from IPython import embed
from sys import exit

class GASLFRDataset(Dataset):

    def __init__(
            self,
            preprocess_cls,
            split_csv_fpath,
            prep_max_seq_len, 
            prep_include_z,
            prep_include_vels,
            char_to_idx_fpath=CHAR_TO_IDX_FPATH,
            sos_idx=PHRASE_SOS_IDX,
            eos_idx=PHRASE_EOS_IDX,
            pad_idx=PHRASE_PAD_VALUE,
            max_phrase_len=MAX_PHRASE_LEN,
            data_root_path=None,
            debug=False
        ):
        super().__init__()
        
        self.max_phrase_len = max_phrase_len
        if isinstance(split_csv_fpath, list):
            self.split_df = []
            for file_path in split_csv_fpath:
                df = pd.read_csv(file_path)
                self.split_df.append(df)
            self.split_df = pd.concat(self.split_df, axis=0, ignore_index=True)
        else:
            self.split_df = pd.read_csv(split_csv_fpath)
        self.split_df.path = self.split_df.path.apply(lambda x: x.replace('train_landmarks_extracted', 'train_landmarks').replace('supplemental_landmarks_extracted', 'supplemental_landmarks').replace('gz', 'parquet'))
        
        if data_root_path:
            self.split_df.path = self.split_df.path.apply(lambda x: x.replace('./', data_root_path))
            char_to_idx_fpath = char_to_idx_fpath.replace('./', data_root_path)
        
        if debug:
            self.split_df = self.split_df[:100]
        
        self.char_to_idx = read_json(char_to_idx_fpath)
        self.char_to_idx['<sos>'] = sos_idx
        self.char_to_idx['<eos>'] = eos_idx
        self.char_to_idx['<pad>'] = pad_idx

        preprocess_cls = import_class(preprocess_cls)
        self.prep_layer = preprocess_cls(
            max_seq_len=prep_max_seq_len,
            include_z=prep_include_z,
            include_vels=prep_include_vels,
        )
        self.idx_to_char = {v:k for k,v in self.char_to_idx.items()}

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        df = self.split_df.iloc[idx]
        seq_idx = df.sequence_id
        
        X = torch.FloatTensor(pd.read_parquet(df.path, columns=['x','y','z'], engine='fastparquet').to_numpy())
        X = X.reshape(-1, MEDIAPIPE_NUM_KEYPOINTS, 3)
        X = self.prep_layer(X)
        X = X.squeeze(0) # Remove batch dimension (needed on inference)
       
        y = torch.LongTensor([self.char_to_idx[c] for c in df.phrase])
        sos = torch.LongTensor([self.char_to_idx['<sos>']])
        eos = torch.LongTensor([self.char_to_idx['<eos>']])
        y = torch.cat((sos, y, eos), axis=-1)
        y = F.pad(y, (0, self.max_phrase_len-len(y)), value=self.char_to_idx['<pad>'])
        
        return seq_idx, X, y
    
if __name__ == '__main__':
    from IPython import embed
    from sys import exit

    d = GASLFRDataset('./data_gen/baseline/train_split.csv', 385, False, False)
    X, y = d[0]
    embed(); exit()
