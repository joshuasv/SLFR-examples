from IPython import embed
from sys import exit

import os
import argparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils import read_yaml, ensure_clean_dir
from src.globals import SEED, TRAIN_CSV_FPATH, SUPPLEMENTAL_CSV_FPATH, TRAIN_DATA_DPATH, SUPPLEMENTAL_DATA_DPATH


def generate_splits(df, val_n_samples, test_n_samples, group_split=False, group_by=None, seed=SEED):
    if group_split:
        test_splitter = GroupShuffleSplit(n_splits=2, test_size=0.1)
        participant_ids = df.participant_id.values
        idxs = list(range(len(df)))
        train_idx,  test_idx = next(test_splitter.split(idxs, groups=participant_ids))
        test_data = df.iloc[test_idx]
        df = df.iloc[train_idx]

        val_splitter = GroupShuffleSplit(n_splits=2, test_size=0.1)
        participant_ids = df.participant_id.values
        idxs = list(range(len(df)))
        train_idx,  val_idx = next(val_splitter.split(idxs, groups=participant_ids))

        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]

        assert not(bool(set(train_data.participant_id).intersection(set(val_data.participant_id)).intersection(set(test_data.participant_id))))

    else:
        test_data = df.sample(n=test_n_samples, random_state=seed)
        test_idxs = test_data.sequence_id
        df = df[~df.sequence_id.isin(test_idxs)]

        val_data = df.sample(n=val_n_samples, random_state=seed)
        val_idxs = val_data.sequence_id
        df = df[~df.sequence_id.isin(val_idxs)]

        test_data = test_data.set_index('sequence_id')
        val_data = val_data.set_index('sequence_id')
        train_data = df.set_index('sequence_id')


    return train_data, val_data, test_data

def add_seq_path(df, root_data_dpath):
    return df.sequence_id.apply(lambda s_id: os.path.join(root_data_dpath, f'{s_id}.gz'))

def filter(df, min_seq_len, max_seq_len):
    if min_seq_len is not None:
        df = df.query('frames > @min_seq_len')

    if max_seq_len is not None:
        df = df.query('frames < @max_seq_len')

    return df

def frame_char_ratio_filter(df, valid_frame_char_ratio):

    if valid_frame_char_ratio is not None:
        df = df.query('frame_char_ratio >= @valid_frame_char_ratio')

    return df


def main(args):
    config = read_yaml(args.config)
    curr_dpath = os.path.split(__file__)[0]
    out_dpath = os.path.join(curr_dpath, config['config_name'])
    if not ensure_clean_dir(out_dpath):
        exit()

    # read train and supplemental data csv
    train_df = pd.read_csv(TRAIN_CSV_FPATH)
    supp_df = pd.read_csv(SUPPLEMENTAL_CSV_FPATH)

    # filter
    train_df = filter(
        df=train_df, 
        min_seq_len=config.get('min_seq_len', None),
        max_seq_len=config.get('max_seq_len', None)
    )

    train_df = frame_char_ratio_filter(
        df=train_df,
        valid_frame_char_ratio=config.get('valid_frame_char_ratio', None)
    )

    train_df.path = add_seq_path(train_df, TRAIN_DATA_DPATH)
    supp_df.path = add_seq_path(supp_df, SUPPLEMENTAL_DATA_DPATH)

    

    train_train_df, train_val_df, train_test_df = generate_splits(
        df=train_df, 
        val_n_samples=config['val_n_samples'], 
        test_n_samples=config['test_n_samples'],
        group_split=config.get('group_split', False),
        group_by=config.get('group_by', None)
    )
    train_train_df.to_csv(os.path.join(out_dpath, 'train_split.csv'))
    train_val_df.to_csv(os.path.join(out_dpath, 'val_split.csv'))
    train_test_df.to_csv(os.path.join(out_dpath, 'test_split.csv'))

    supp_train_df, supp_val_df, supp_test_df = generate_splits(
        df=supp_df, 
        val_n_samples=config['val_n_samples'], 
        test_n_samples=config['test_n_samples']
    )
    supp_train_df.to_csv(os.path.join(out_dpath, 'supp_train_split.csv'))
    supp_val_df.to_csv(os.path.join(out_dpath, 'supp_val_split.csv'))
    supp_test_df.to_csv(os.path.join(out_dpath, 'supp_test_split.csv'))


if __name__ == '__main__':
    # python -m data_gen.generate_splits --config ./data_gen/config/baseline.yaml 

    parser = argparse.ArgumentParser('Generate train, validation, and test splits!')
    parser.add_argument(
        '--config',
        type=str
    )

    args = parser.parse_args()
    main(args)