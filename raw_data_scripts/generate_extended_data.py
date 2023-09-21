import os
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


class Cfg:
    RANDOM_STATE = 2023
    INPUT_ROOT = Path('/home/gts/projects/jsoutelo/GASLFR_raw')
    OUTPUT_ROOT = Path('/home/gts/projects/jsoutelo/GASLFR')
    INDEX_MAP_FILE = INPUT_ROOT / 'character_to_prediction_index.json'
    TRAN_FILE = INPUT_ROOT / 'train.csv'
    META_FILE = INPUT_ROOT / 'supplemental_metadata.csv'
    INDEX = 'sequence_id'
    ROW_ID = 'row_id'


def read_index_map(file_path=Cfg.INDEX_MAP_FILE):
    """Reads the sign to predict as json file."""
    with open(file_path, "r") as f:
        result = json.load(f)
    return result    

def read_train(file_path=Cfg.TRAN_FILE):
    """Reads the train csv as pandas data frame."""
    return pd.read_csv(file_path).set_index(Cfg.INDEX)

def read_supplemental_metadata(file_path=Cfg.META_FILE):
    """Reads the supplemental metadata csv as a pandas data frame."""
    return pd.read_csv(file_path).set_index(Cfg.INDEX)

def read_data(file_path, input_root=Cfg.INPUT_ROOT):
    """Reads landmak data by the given file path."""
    data = pd.read_parquet(input_root / file_path)
    return data

def read_landmark_data_by_id(sequence_id, train_data):
    """Reads the landmark data by the given sequence id."""
    file_path = train_data.loc[sequence_id]['path']
    
    all_data = read_data(file_path)
    
    sequence_data = all_data.loc[[sequence_id]]
    sequence_data = format_landmark_data(sequence_data)
    
    return sequence_data

def format_landmark_data(sequence_data_df):
    x_df, y_df, z_df = np.array_split(sequence_data_df, indices_or_sections=3, axis=1)

    x_df = pd.melt(x_df, id_vars=['frame'], var_name='landmark_index', value_name='x')
    y_df = pd.melt(y_df, var_name='landmark_index', value_name='y')
    z_df = pd.melt(z_df, var_name='landmark_index', value_name='z')

    x_df['y'] = y_df.y
    x_df['z'] = z_df.z
    x_df.frame = x_df.frame.astype(int)

    x_df['type'] = x_df.landmark_index.apply(lambda x: '_'.join(x.split('_')[1:-1]))
    x_df.landmark_index = x_df.landmark_index.apply(lambda x: x.split('_')[-1])

    return x_df

from functools import partial
from multiprocessing import Pool

def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(tqdm(pool.imap_unordered(func, data_split), total=len(data_split)))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


def a_function(row, df):
    frames = len(read_landmark_data_by_id(row.name, df).frame.unique())
    row['frames'] = frames
    return row

if __name__ == '__main__':

    train_df = read_train()
    # train_df = train_df[:1000]
    
    # 1000 examples, 17min
    # tqdm.pandas(desc='train_df')
    # train_df['frames'] = train_df.progress_apply(lambda row: len(read_landmark_data_by_id(row.name, train_df).frame.unique()), axis=1)
    # train_df.to_csv(Cfg.OUTPUT_ROOT / 'etrain.csv')

    # 1000 examples, 6min
    # total time, 4:26:35
    new_df = parallelize_on_rows(train_df, partial(a_function, df=train_df))
    new_df.to_csv(Cfg.OUTPUT_ROOT / 'etrain.csv')

    del train_df
    del new_df
    gc.collect()

    # Drop the buggy index
    buggy_meta_idx = 435344989
    meta_df = read_supplemental_metadata()
    meta_df = meta_df.drop(index=buggy_meta_idx, axis=0)

    new_df = parallelize_on_rows(meta_df, partial(a_function, df=meta_df))
    new_df.to_csv(Cfg.OUTPUT_ROOT / 'esupplemental_metadata.csv')