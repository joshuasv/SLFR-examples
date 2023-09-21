import os
import glob
import time
import argparse
from shutil import rmtree
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
from tqdm import tqdm

from IPython import embed; from sys import exit

def extract_embedded_sequences(fpath, out_dpath, i, exclude):
    
    df = pd.read_parquet(fpath)

    if exclude and os.path.basename(fpath) == '1249944812.parquet': # Does not exist, known bug
        df = df.drop([df.sequence_id == 435344989].index)

    sequences_idxs = df.index.unique().to_list()
    # iterate over all sequences
    for sequence_id in sequences_idxs:
        # extract sequence from all data
        seq_df = df.loc[[sequence_id]]
        # sort by frame in ascending order
        seq_df = seq_df.sort_values(by='frame')
        # format data
        # columns to rows
        melt_df = pd.melt(seq_df, id_vars='frame')
        # add type column [face, right_hand, pose...]
        melt_df['type'] = melt_df.variable.apply(lambda x: '_'.join(x.split('_')[1:-1]))
        # add landmark index column [face landmark 0, face landmark 1, ...]
        melt_df['landmark_index'] = melt_df.variable.apply(lambda x: int(x.split('_')[-1]))
        # create final format df based on the unique values of frame+type+landmark_index
        new_df = pd.DataFrame(melt_df.groupby(['frame', 'type', 'landmark_index']).size().reset_index(level=['frame', 'type', 'landmark_index']))
        # sort values first all from face, then left hand, then pose, lastly right hand
        new_df = new_df.sort_values(by=['frame', 'type', 'landmark_index'])
        # split x, y, and z values in its own dataframe, not taking into account frame column, hence [:, 1:]
        xyz_split = np.array_split(seq_df.iloc[:, 1:], 3, axis=1)
        # add x, y, and z values to formated df
        new_df['x'] = xyz_split[0].to_numpy().flatten()
        new_df['y'] = xyz_split[1].to_numpy().flatten()
        new_df['z'] = xyz_split[2].to_numpy().flatten()
        # remove column named 0
        new_df = new_df.drop(columns=0)
        # store formated sequence to disk
        sequence_fname = f"{sequence_id}.parquet"
        sequence_fpath = os.path.join(out_dpath, sequence_fname)
        new_df.to_parquet(sequence_fpath, engine='fastparquet', index=None, partition_cols=None, compression=None)

def main(args):
    if os.path.exists(args.out_dpath) and len(os.listdir(args.out_dpath)) > 0:
        user_inp = input(f"{args.out_dpath} already exits. Remove? [Y/n]: ")
        if user_inp.lower() == 'y' or user_inp == '':
            rmtree(args.out_dpath)
    os.makedirs(args.out_dpath, exist_ok=True)

    fpaths = glob.glob(os.path.join(args.data_dpath, '*'))
    start = time.perf_counter()
    with tqdm(total=len(fpaths), position=0, leave=True) as pbar:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for i, fpath in enumerate(fpaths):
                future = executor.submit(extract_embedded_sequences, fpath, args.out_dpath, i, args.exclude)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)
            wait(futures)
    elapsed = start - time.perf_counter()
    print(f"Took {timedelta(seconds=elapsed)}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Extract all individual sequences from GASLFR.')

    parser.add_argument(
        '--data-dpath',
        type=str,
        default='/home/gts/projects/jsoutelo/GASLFR_raw/train_landmarks'
    )

    parser.add_argument(
        '--out-dpath',
        type=str,
        default='./train_landmarks_extracted'
    )

    parser.add_argument(
        '--exclude',
        action='store_true'
    )
    args = parser.parse_args()

    main(args)