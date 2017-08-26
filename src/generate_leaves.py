from utils import *

import argparse
import pandas as pd
import scipy as sp

import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

SEED = 12313

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path', type=str)
parser.add_argument('te_src_path', type=str)
parser.add_argument('tr_dst_path', type=str)
parser.add_argument('te_dst_path', type=str)
parser.add_argument('is_fold', type=str)

args = vars(parser.parse_args())

FIELDS = ['hour', 'countrycode', 'browserid', 'devid']

def encode_labels(tr, te, features):
    for f in features:
        lbl = LabelEncoder()
        lbl.fit(tr.loc[:, f])
        
        tr.loc[:, f] = lbl.transform(tr.loc[:, f])
        te.loc[:, f] = lbl.transform(te.loc[:, f])
      
    return tr, te


def get_leaves(tr_src_path, te_src_path, tr_dst_path, te_dst_path, is_fold):
    train = pd.read_csv(tr_src_path)
    test  = pd.read_csv(te_src_path)
    
    # sparse representation
    train_sparse, test_sparse = prepare_sparse_representation(train, test)
    
    
    n_estimators = 30

    xgb_pars = {
        'eta': 0.3,
        'max_depth': 7,
        'gamma': 1,
        'min_child_weight': 1,
        'colsample_bytree': .8,
        'colsample_bylevel': .8,
        'subsample': 1.,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'nthread': 12,
        'seed': SEED
    }
    
    
    if is_fold:
        dtrain = xgb.DMatrix(train_sparse, train.loc[:, 'click'])
        dval   = xgb.DMatrix(test_sparse)
    
        model0  = xgb.train(xgb_pars, dtrain, num_boost_round=n_estimators)
        leaves1 = model0.predict(dval, pred_leaf=True).astype('uint8')

        dtrain = xgb.DMatrix(test_sparse, test.loc[:, 'click'])
        dval   = xgb.DMatrix(train_sparse)

        model1  = xgb.train(xgb_pars, dtrain, num_boost_round=n_estimators)
        leaves0 = model1.predict(dval, pred_leaf=True).astype('uint8')
        
        np.save(tr_dst_path, leaves0)
        np.save(te_dst_path, leaves1)
        
    else:
        dtrain = xgb.DMatrix(train_sparse, train.loc[:, 'click'])
        dval   = xgb.DMatrix(test_sparse)
        
        model       = xgb.train(xgb_pars, dtrain, num_boost_round=n_estimators)
        leaves_test = model.predict(dval, pred_leaf=True).astype('uint8')
        
        # train is vertical stack of fold0 and fold1
        leaves0 = np.load('../data/folds/leaves/fold0_leaves.npy')
        leaves1 = np.load('../data/folds/leaves/fold1_leaves.npy')
        
        leaves_train = np.vstack((leaves0, leaves1))
        
        np.save(tr_dst_path, leaves_train)
        np.save(te_dst_path, leaves_test)
    
get_leaves(args['tr_src_path'],
           args['te_src_path'],
           args['tr_dst_path'],
           args['te_dst_path'],
           args['is_fold'] == 'y'
          )

