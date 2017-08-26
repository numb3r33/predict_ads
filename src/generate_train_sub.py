#!/usr/bin/env python3

import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path', type=str)
parser.add_argument('te_src_path', type=str)
parser.add_argument('tr_dst_path', type=str)

args = vars(parser.parse_args())

def get_train_sub(tr_src_path, te_src_path, tr_dst_path):
    train, test = load_data(tr_src_path, te_src_path)
    train_sub   = train_in_test(train, test)
    #train_sub   = last_3_days(train)
    
    train_sub.to_csv(tr_dst_path, index=False)
    
    
get_train_sub(args['tr_src_path'], args['te_src_path'], args['tr_dst_path'])