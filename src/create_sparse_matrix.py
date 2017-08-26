import argparse
import pandas as pd
import scipy as sp
import time

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path', type=str)
parser.add_argument('te_src_path', type=str)
parser.add_argument('tr_dst_path', type=str)
parser.add_argument('te_dst_path', type=str)

args = vars(parser.parse_args())

def prepare_sparse_matrix(tr_src_path, te_src_path, tr_dst_path, te_dst_path):
    train = pd.read_csv(tr_src_path)
    test  = pd.read_csv(te_src_path)
    
    st    = time.clock()
    
    print('Preparing sparse representation') 
    train_sparse, test_sparse = prepare_sparse_representation(train, test)
    
    print('Took: {} seconds'.format(time.clock() - st))
    
    print('Save matrices')
    sp.sparse.save_npz(tr_dst_path, train_sparse)
    sp.sparse.save_npz(te_dst_path, test_sparse)
    
    print('Took: {} seconds'.format(time.clock() - st))
    
prepare_sparse_matrix(args['tr_src_path'],
                      args['te_src_path'],
                      args['tr_dst_path'],
                      args['te_dst_path']
                     )