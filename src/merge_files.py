import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path', type=str)
parser.add_argument('te_src_path', type=str)
parser.add_argument('tr_dst_path', type=str)

args = vars(parser.parse_args())

def merge_files(tr_src_path, te_src_path, tr_dst_path):
    fold0 = pd.read_csv(tr_src_path)
    fold1 = pd.read_csv(te_src_path)
    
    print('Merge and save')
    pd.concat((fold0, fold1)).to_csv(tr_dst_path, index=False)
    
    
merge_files(args['tr_src_path'],
       args['te_src_path'],
       args['tr_dst_path'])

