#!/usr/bin/env python3

"""
This module splits the training dataset into train and validation folds of equal size.
"""

import argparse
import numpy as np

from sklearn.model_selection import KFold

from utils import *

SEED = 123
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path', type=str)
parser.add_argument('te_src_path', type=str)
parser.add_argument('fold0_dst_path', type=str)
parser.add_argument('fold1_dst_path', type=str)
parser.add_argument('tr_sub', type=str)

args = vars(parser.parse_args())

def split_datasets(tr_src_path, te_src_path, tr_sub, fold0_dst_path, fold1_dst_path):
    print('Loading Dataset')
    train, test = load_data(tr_src_path, te_src_path)
    print('Dataset loaded successfully')
    
    if tr_sub == 't':
        print('Creating subset of training data based on siteid')
        train = train_in_test(train, test)
        print('Subset created successfully')
    elif tr_sub == 'o':
        print('Creating subset of training data based on offerid')
        train = train_in_test_offer(train, test)
        print('Subset created successfully')
    elif tr_sub == 'l':
        print('Creating subset based on last 3 days')
        train = last_3_days(train)
    elif tr_sub == 's':
        print('Creating subset based on same weekdays')
        train = same_weekday(train)
        
    kf = KFold(n_splits=2, shuffle=True, random_state=SEED)
    itr, ite = next(kf.split(train))
    
    print('Creating two folds out of training set')
    train.iloc[itr].to_csv(fold0_dst_path, index=False)
    train.iloc[ite].to_csv(fold1_dst_path, index=False)
    print('Folds saved successfully')
        

    
split_datasets(args['tr_src_path'],
               args['te_src_path'],
               args['tr_sub'],
               args['fold0_dst_path'],
               args['fold1_dst_path']
              )
        