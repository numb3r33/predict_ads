import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

parser = argparse.ArgumentParser()

parser.add_argument('--src_path', type=str)
parser.add_argument('--pred_paths', type=str, nargs='+')
parser.add_argument('--weights', type=str, nargs='+')
parser.add_argument('--method', type=str)
parser.add_argument('--dst_path', type=str, default='')


args = vars(parser.parse_args())

def ensemble_predictions(pred_paths, weights):
    preds = []
    
    for pred_path in pred_paths:
        preds.append(rankdata(pd.read_csv(pred_path, header=None)[0]))
    
    weights  = np.array(weights)
    weights  = weights.astype(np.float)
    ensemble_preds =  np.average(preds, axis=0, weights=weights)
    
    return ensemble_preds

def check_predictions(src_path, pred_paths, weights):
    
    true  = pd.read_csv(src_path, usecols=['click'])
    ensemble_preds = ensemble_predictions(pred_paths, weights)
    
    print('Ensemble pred: {}'.format(roc_auc_score(true, ensemble_preds)))   

def save_predictions(src_path, pred_paths, weights, dst_path):
    true           = pd.read_csv(src_path, usecols=['ID'])
    ensemble_preds = ensemble_predictions(pred_paths, weights)
    
    sub = pd.DataFrame({'ID': true.ID, 'click': ensemble_preds})
    sub.to_csv('%s'%(dst_path), index=False)
    
if args['method'] == 'check':
    check_predictions(args['src_path'], args['pred_paths'], args['weights'])
else:
    save_predictions(args['src_path'], args['pred_paths'], args['weights'], args['dst_path'])