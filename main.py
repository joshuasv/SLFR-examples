import os
import random
import argparse

import yaml
import torch
import numpy as np
from Trainer import Trainer


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_parser():
    parser = argparse.ArgumentParser(description='examples')
    # General flags
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode; default false')
    parser.add_argument(
        '-y',
        action='store_true',
        help='Debug mode; default false')
    parser.add_argument(
        '--phase',
        default='train',
        help='must be train or test')
    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    parser.add_argument(
        '--work-dir',
        type=str,
        required=False,
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        type=str,
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--print-log',
        action='store_true',
        help='print logging or not')
    # Feeder flags
    parser.add_argument(
        '--feeder',
        type=str,
        default='src.feeders.GASLFRDataset.GASLFRDataset',
        help='data loader will be used')
    parser.add_argument(
        '--train_feeder_args',
        default=dict(),
        help='arguments for the training feeder')
    parser.add_argument(
        '--test_feeder_args',
        default=dict(),
        help='arguments for the testing feeder')
    # Model flags
    parser.add_argument(
        '--model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--encoder',
        default=None,
        help='the encoder will be used')
    parser.add_argument(
        '--decoder',
        default=None,
        help='the decoder will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of the model')
    parser.add_argument(
        '--encoder-args',
        type=dict,
        default=dict(),
        help='the arguments of the encoder model')
    parser.add_argument(
        '--decoder-args',
        type=dict,
        default=dict(),
        help='the arguments of the decoder model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    # Optimizer flags
    parser.add_argument(
        '--optimizer',
        type=str,
        default='torch.optim.AdamW',
        help='the optimizer class to be used during training'
    )
    parser.add_argument(
        '--optimizer-args',
        default=dict(),
        help='the optimizer arguments'
    )
    # Loss flags
    parser.add_argument(
        '--loss_args',
        default=dict(),
        help='the loss arguments to be used during training'
    )
    # Learning rate scheduler flags
    parser.add_argument(
        '--lr-scheduler',
        type=str,
        default=None,
        help='learning rate scheduler class')
    parser.add_argument(
        '--lr-scheduler-args',
        type=dict,
        default=dict(),
        help='learning rate scheduler arguments')
    # Training related flags
    parser.add_argument(
        '--base_lr',
        type=float,
        default=1e-4,
        help='learning rate to be used during training')
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=100,
        help='number of epochs to train')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='training batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=256,
        help='test batch size')
    parser.add_argument(
        '--forward-batch-size',
        type=int,
        default=16,
        help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='path of previously saved training checkpoint')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--run-eval',
        action='store_true',
        help='wether to run evaluation altogether or not')
    parser.add_argument(
        '--display-predictions-interval',
        type=int,
        default=1,
        help='the interval for displaying some model predictions (#iteration)')
    
    return parser

def main():
    parser = get_parser()
    # Load args from config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, yaml.SafeLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)
    args = parser.parse_args()
    init_seed(args.seed)
    trainer = Trainer(args)
    trainer.start()

if __name__ == '__main__':
    main()