from utils import *

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path', type=str)
parser.add_argument('te_src_path', type=str)
parser.add_argument('tr_dst_path', type=str)
parser.add_argument('te_dst_path', type=str)
parser.add_argument('remove_uncommon', type=str)

args = vars(parser.parse_args())

def load_data(tr_src_path, te_src_path):
    train = pd.read_csv(tr_src_path, parse_dates=['datetime'])
    test  = pd.read_csv(te_src_path, parse_dates=['datetime'])
    
    return train, test

def process_data(tr_src_path, te_src_path, tr_dst_path, te_dst_path, remove_uncommon):
    """
    This function does preprocessing of data
    
    1. Fill missing value.
    2. Merge levels of categorical data.
    3. Time based features.
    """
    
    train, test = load_data(tr_src_path, te_src_path)
    
    print('Cleaning browser ids')
    train, test = clean_browserids(train, test)
    
    if remove_uncommon:
        print('Replace uncommon features with 9999')
        train, test = replace_uncommon_levels(train, test, ['siteid', 'offerid', 'merchant'])
    
    print('Fill missing values for browser ids')
    train, test = fill_missing_values(train, test, 'browserid', 'missing_browser')
    
    print('Fill missing values for device id')
    
    train.loc[train.browserid == 'Edge', 'devid']   = 'Tablet'
    train.loc[train.browserid == 'Opera', 'devid']  = 'Mobile'
    train.loc[train.browserid == 'Safari', 'devid'] = 'Tablet'
    
    test.loc[test.browserid == 'Edge', 'devid']   = 'Tablet'
    test.loc[test.browserid == 'Opera', 'devid']  = 'Mobile'
    test.loc[test.browserid == 'Safari', 'devid'] = 'Tablet'
    
    train, test = fill_missing_values(train, test, 'devid', 'missing_dev')
    
    print('Fill missing values for site id')
    train, test = fill_missing_values(train, test, 'siteid', 9999999)
    
    print('Create hour based features')
    train, test = create_time_features(train, test)
    
    print('Saving processed data')
    train.to_csv(tr_dst_path, index=False)
    test.to_csv(te_dst_path, index=False)


process_data(args['tr_src_path'],
             args['te_src_path'],
             args['tr_dst_path'],
             args['te_dst_path'],
             args['remove_uncommon'] == 'y'
            )