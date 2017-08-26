"""
This file contains all the utility functions.
"""
import pandas as pd
import scipy as sp
import hashlib

NR_BINS = 1000000


def load_data(tr_path, te_path):
    """
    Loads dataset
    
    Arguments:
    ----------
    tr_path: string
    te_path: string
    
    Return:
    -------
    train: dataframe
    test : dataframe or None
    """
    
    dtype = {
        'siteid': 'float32',
        'offerid': 'uint32',
        'category': 'uint32',
        'merchant': 'uint32',
        'click': 'uint8'
    }
    
    # mention dtypes to save some memory
    train = pd.read_csv(tr_path, dtype=dtype, parse_dates=['datetime'])
    test  = None
    
    if te_path is not None:
        del dtype['click']
        test  = pd.read_csv(te_path, dtype=dtype, parse_dates=['datetime'])
    
    return train, test

def train_in_test(train, test):
    """
    This function return only those instances where siteid in train
    is present in the test set as well.
    
    Arguments:
    ----------
    
    train: dataframe
    test : dataframe
    
    Return:
    -------
    
    train_sub: Subset of training set
    """
    
    return train.loc[test.siteid.isin(train.siteid).values, :]

def last_3_days(train):
    """
    This function return only instances for last 3 days in the training set.
    
    Arguments:
    ------------
    
    train: dataframe
    test : dataframe
    
    Return:
    -------
    
    train_sub: Subset of training set
    """
    train.loc[:, 'datetime'] = pd.to_datetime(train.datetime)
    train.loc[:, 'day']      = train.datetime.dt.day
    
    return train.loc[train.day.isin([18, 19, 20]), :]

def same_weekday(train):
    """
    This function return instances for same weekdays as in test set.
    
    Arguments:
    --------------
    
    train: dataframe
    test : dataframe
    
    Return:
    --------------
    
    train_sub: Subset of training set
    """
    
    train.loc[:, 'datetime'] = pd.to_datetime(train.datetime)
    train.loc[:, 'day']      = train.datetime.dt.day
    
    return train.loc[train.day.isin([14, 15, 16]), :]
    


def train_in_test_offer(train, test):
    """
    This function return only those instances where offerid in train
    is present in the test set as well.
    
    Arguments:
    -------------
    
    train: dataframe
    test : dataframe
    
    Return:
    --------------
    
    train_sub: Subset of training set
    """
    
    return train.loc[test.offerid.isin(train.offerid).values, :]
    

def clean_browserids(train, test):
    """
    Cleans up browser names. There are multiple levels for this categorical variable
    but they in fact represent the same browser.
    
    Arguments:
    ----------
    
    train: dataframe
    test : dataframe
    
    Return:
    ----------
    
    train_cleaned: dataframe
    test_cleaned : dataframe
    
    """
    
    # map browser names
    replacement = {
    'Chrome': 'Google Chrome',
    'IE'    : 'Internet Explorer',
    'InternetExplorer': 'Internet Explorer',
    'Firefox': 'Mozilla Firefox',
    'Mozilla': 'Mozilla Firefox'
    }

    train.loc[:, 'browserid'] = train.browserid.replace(replacement)
    test.loc[:, 'browserid']  = test.browserid.replace(replacement)
    
    return train, test

def fill_missing_values(train, test, feature_name, missing_val):
    """
    This function helps fill missing values for a certain feature 
    with a missing value provided.
    
    Arguments:
    -----------
    
    train: dataframe
    test : dataframe
    feature_name: string
    missing_val : any data type
    
    Return:
    -----------
    train_imputed : dataframe
    test_imputed  : dataframe
    
    """
    
    train.loc[:, feature_name] = train[feature_name].fillna(missing_val)
    test.loc[:, feature_name]  = test[feature_name].fillna(missing_val)
    
    return train, test


def create_time_features(train, test):
    """
    This function decomposes date into hour
    
    Arguments:
    ----------
    
    train: dataframe
    test : dataframe
    
    Return:
    ---------
    train_with_hour: dataframe
    test_with_hour : dataframe
    """
    
    train = train.assign(hour=train.datetime.dt.hour)
    test  = test.assign(hour=test.datetime.dt.hour)
    
    return train, test

def replace_uncommon_levels(df_train, df_test, features):
    """
    This function replaces levels of categorical features which are present in train
    and not in test and vice-versa.
    
    Arguments:
    ----------
    
    df_train: dataframe
    df_test : dataframe
    features: list of categorical features
    
    Return:
    --------
    df_train_common: dataframe
    df_test_common : dataframe
    """
    
    for c in features:
        common = set(df_train[c].unique()) & set(df_test[c].unique())
        df_train.loc[~df_train[c].isin(common), c] = -99
        df_test.loc[~df_test[c].isin(common), c] = -99
        
    return df_train, df_test

def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)

def prepare_sparse_representation(train, test):
    """
    This function creates sparse representation of four features
    train, browserid, devid and country code
    
    Arguments:
    -----------
    train: dataframe
    test : dataframe
    
    Return:
    ------------
    
    train_sparse: Sparse matrix
    test_sparse : Sparse matrix
    
    """
    hour_sparse    = pd.get_dummies(train.hour, prefix='hour', drop_first=True, sparse=True)
    browser_sparse = pd.get_dummies(train.browserid, prefix='browser', drop_first=True, sparse=True)
    dev_sparse     = pd.get_dummies(train.devid, prefix='device', drop_first=True, sparse=True)
    cc_sparse      = pd.get_dummies(train.countrycode, prefix='cc', drop_first=True, sparse=True)
    
    
    train_sparse   = sp.sparse.hstack((hour_sparse,
                                       browser_sparse,
                                       dev_sparse,
                                       cc_sparse
                                      ))
    
    hour_sparse    = pd.get_dummies(test.hour, prefix='hour', drop_first=True, sparse=True)
    browser_sparse = pd.get_dummies(test.browserid, prefix='browser', drop_first=True, sparse=True)
    dev_sparse     = pd.get_dummies(test.devid, prefix='device', drop_first=True, sparse=True)
    cc_sparse      = pd.get_dummies(test.countrycode, prefix='cc', drop_first=True, sparse=True)
    
    
    test_sparse   = sp.sparse.hstack((hour_sparse,
                                      browser_sparse,
                                      dev_sparse,
                                      cc_sparse
                                      ))
    
    return train_sparse, test_sparse


def mean_target_value(df, features, target, C):
    """
    features: list of features
    target  : target label
    """
    K      = df.groupby(features).size()
    mean_y = df.groupby(features)[target].mean()
    global_mean_y = df[target].mean()
    return (mean_y * K + global_mean_y * C) / ( K + C )

def prepare_mtv_features(train, test, features):
    for f in features:
        print('Feature: {}'.format(f))
        
        f_mtv = mean_target_value(train, [f], 'click', C=10)
        train.loc[:, f] = train.loc[:, f].map(f_mtv)
        test.loc[:, f]  = test.loc[:, f].map(f_mtv)
    
    return train, test





        
  