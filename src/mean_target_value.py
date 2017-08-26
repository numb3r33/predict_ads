import argparse
import pandas as pd
import time

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path', type=str)
parser.add_argument('te_src_path', type=str)
parser.add_argument('tr_dst_path', type=str)
parser.add_argument('te_dst_path', type=str)

args = vars(parser.parse_args())

def prepare_mtv_matrix(tr_src_path, te_src_path, tr_dst_path, te_dst_path):
    train = pd.read_csv(tr_src_path)
    test  = pd.read_csv(te_src_path)
    
    st    = time.clock()
    
    print('Preparing sparse representation') 
    train_mtv, test_mtv = prepare_mtv_features(train, test, ['hour', 'browserid', 'devid', 'countrycode'])
    
    print('Took: {} seconds'.format(time.clock() - st))
    
    print('Save matrices')
    train_mtv.to_csv(tr_dst_path, index=False)
    test_mtv.to_csv(te_dst_path, index=False)
    
    print('Took: {} seconds'.format(time.clock() - st))
    
prepare_mtv_matrix(args['tr_src_path'],
                   args['te_src_path'],
                   args['tr_dst_path'],
                   args['te_dst_path']
                  )