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
parser.add_argument('train_method', type=str)
parser.add_argument('is_fold', type=str)

args = vars(parser.parse_args())

def train(tr_src_path, te_src_path, tr_dst_path, te_dst_path, train_method, is_fold):
    if train_method == 's':
        train  = sp.sparse.load_npz(tr_src_path)
        test   = sp.sparse.load_npz(te_src_path)
        
        print('Shape of train', train.shape)
        print('Shape of test', test.shape)
        
    elif train_method == 'm':    
        train = pd.read_csv(tr_src_path, usecols=['hour', 'browserid', 'devid', 'countrycode'])
        test  = pd.read_csv(te_src_path, usecols=['hour', 'browserid', 'devid', 'countrycode'])
        
        print('Shape of train ', train.shape)
        print('Shape of test ', test.shape)
    else:
        train = pd.read_csv(tr_src_path, usecols=['hour', 
                                                  'browserid', 
                                                  'devid', 
                                                  'countrycode',
                                                  'siteid_count',
                                                  'offerid_count'
                                                 ])
        
        test  = pd.read_csv(te_src_path, usecols=['hour', 
                                                  'browserid', 
                                                  'devid', 
                                                  'countrycode',
                                                  'siteid_count',
                                                  'offerid_count'
                                                 ])
        
        print('Shape of train ', train.shape)
        print('Shape of test ', test.shape)
    
    n_estimators = 100

    xgb_pars = {
        'eta': 0.1,
        'max_depth': 8,
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
        # not very optimal way to do this but going with this
        ytrain = pd.read_csv('../data/folds/fold0.csv', usecols=['click'])
        ytest  = pd.read_csv('../data/folds/fold1.csv', usecols=['click'])
        
        print('Training model on fold0')
        dtrain = xgb.DMatrix(train, ytrain)
        dval   = xgb.DMatrix(test)
        
        model0  = xgb.train(xgb_pars, dtrain, num_boost_round=n_estimators)
        preds1  = model0.predict(dval)
        
        print('Training model on fold1')
        dtrain = xgb.DMatrix(test, ytest)
        dval   = xgb.DMatrix(train)

        model1  = xgb.train(xgb_pars, dtrain, num_boost_round=n_estimators)
        preds0  = model1.predict(dval)
        
        print('Generating predictions')
        np.savetxt(tr_dst_path, preds0)
        np.savetxt(te_dst_path, preds1)
        
        
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
    
train(args['tr_src_path'],
      args['te_src_path'],
      args['tr_dst_path'],
      args['te_dst_path'],
      args['train_method'],
      args['is_fold'] == 'y'
     )

