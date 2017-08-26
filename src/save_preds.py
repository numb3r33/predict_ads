#!/usr/env/bin python3

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('test_path', type=str)
parser.add_argument('pred_path', type=str)
parser.add_argument('dest_path', type=str)

args = vars(parser.parse_args())

def create_submission(test_path, pred_path, dest_path):
    test        = pd.read_csv(test_path, usecols=['ID'])
    final_preds = pd.read_csv(pred_path, header=None)
    
    sub = pd.DataFrame({'ID': test.ID, 'click': final_preds[0]})
    sub.to_csv(dest_path, index=False)
    
    
create_submission(args['test_path'],
                  args['pred_path'],
                  args['dest_path']
                 )