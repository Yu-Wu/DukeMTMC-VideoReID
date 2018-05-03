from __future__ import print_function, absolute_import
from baseline import *
from reid import datasets
from reid import models
import numpy as np
import torch
import argparse
import os

from reid.utils.logging import Logger
import os.path as osp
import sys
from torch.backends import cudnn
from reid.utils.serialization import load_checkpoint
from torch import nn
import time
import pickle


def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True
    save_path = args.logs_dir
    sys.stdout = Logger(osp.join(args.logs_dir, 'log'+ time.strftime(".%m_%d_%H:%M:%S") + '.txt'))

    # get the dataset
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    num_all_examples = len(dataset_all.train)
    
    # get the baseline model
    baseline = Baseline(model_name=args.arch, batch_size=args.batch_size, num_classes=dataset_all.num_train_ids, 
            data_dir=dataset_all.images_dir, save_path=args.logs_dir, max_frames=args.max_frames)

    # train the model 
    baseline.train(dataset_all.train, epochs=70, step_size=55, init_lr=0.1) 

    # evaluate
    baseline.evaluate(dataset_all.query, dataset_all.gallery)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline code for DukeMTMC-VideoReID')
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmc_videoReID',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',
                        choices=models.names())
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    parser.add_argument('--max_frames', type=int, default=900)
    main(parser.parse_args())
