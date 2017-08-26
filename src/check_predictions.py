import argparse
import pandas as pd

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument('src_path', type=str)
parser.add_argument('pred_path', type=str)

args = vars(parser.parse_args())

def check_predictions(src_path, pred_path):
    preds = pd.read_csv(pred_path, header=None)
    true  = pd.read_csv(src_path, usecols=['click'])
    
    print('ROC AUC: {}'.format(roc_auc_score(true, preds[0])))


check_predictions(args['src_path'], args['pred_path'])