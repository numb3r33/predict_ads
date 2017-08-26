import argparse
import collections
import time
import csv

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path', type=str)
parser.add_argument('te_src_path', type=str)
parser.add_argument('tr_dst_path', type=str)
parser.add_argument('te_dst_path', type=str)
parser.add_argument('tr_is_train', type=str)
parser.add_argument('te_is_train', type=str)
parser.add_argument('feature_type',type=str)

args = vars(parser.parse_args())

if args['feature_type'] == 'all':
    FIELDS     = ['hour','browserid', 'devid', 'countrycode',
                  'click', 'siteid', 'offerid', 'merchant', 'category']

    NEW_FIELDS = FIELDS + ['siteid_count', 'offerid_count', 'category_count', 'merchant_count', 'hour_count',
                           'browser_count', 'devid_count', 'country_count', 'site_category_bag'
                          ]
    

elif args['feature_type'] == 'site':
    FIELDS     = ['hour','browserid', 'devid', 'countrycode',
                  'click', 'siteid', 'category', 'merchant']

    NEW_FIELDS = FIELDS + ['siteid_count', 'category_count', 'merchant_count', 'hour_count',
                           'browser_count', 'devid_count', 'country_count', 'site_category_bag'
                          ]
    
elif args['feature_type'] == 'offer':
    FIELDS     = ['hour','browserid', 'devid', 'countrycode',
                  'click', 'offerid', 'category', 'merchant']

    NEW_FIELDS = FIELDS + ['offerid_count', 'category_count', 'merchant_count', 'hour_count',
                           'browser_count', 'devid_count', 'country_count', 'site_category_bag'
                          ]
    
else:
    FIELDS     = ['hour','browserid', 'devid', 'countrycode',
                  'click', 'category', 'merchant']

    NEW_FIELDS = FIELDS + ['category_count', 'merchant_count', 'hour_count',
                           'browser_count', 'devid_count', 'country_count', 'site_category_bag'
                          ]

siteid_cnt        = collections.defaultdict(int)
offerid_cnt       = collections.defaultdict(int)
category_cnt      = collections.defaultdict(int)
merchant_cnt      = collections.defaultdict(int)
hour_cnt          = collections.defaultdict(int)
browser_cnt       = collections.defaultdict(int)
devid_cnt         = collections.defaultdict(int)
country_cnt       = collections.defaultdict(int)
site_category_bag = collections.defaultdict(list) 


start = time.clock()

def scan(path, feature_type):
    for i, row in enumerate(csv.DictReader(open(path)), start=1):
        if i % 1000000 == 0:
            print('{0:6.0f}    {1}m\n'.format(time.clock()-start,int(i/1000000)))
        
        if feature_type == 'all':
            siteid_cnt[row['siteid']]       += 1
            offerid_cnt[row['offerid']]     += 1
            merchant_cnt[row['merchant']]   += 1
            
        elif feature_type == 'site':
            siteid_cnt[row['siteid']]     += 1
        elif feature_type == 'offer':
            offerid_cnt[row['offerid']]   += 1
        else:
            merchant_cnt[row['merchant']]   += 1
            
        category_cnt[row['category']]   += 1
        hour_cnt[row['hour']]           += 1
        browser_cnt[row['browserid']]   += 1
        devid_cnt[row['devid']]         += 1
        country_cnt[row['countrycode']] += 1
        
        site_category_bag[row['siteid']].append(row['category'])
        
        

def gen_data(src_path, dst_path, is_train, feature_type):
    reader = csv.DictReader(open(src_path))
    writer = csv.DictWriter(open(dst_path, 'w'), NEW_FIELDS)
    
    writer.writeheader()
    
    for i, row in enumerate(reader, start=1):
        if i % 1000000 == 0:
            print('{0:6.0f}    {1}m\n'.format(time.clock()-start,int(i/1000000)))
            
        new_row = {}
        for field in FIELDS:
            if not is_train and field == 'click':
                continue
            
            new_row[field] = row[field]
        
        if feature_type == 'all':
            new_row['siteid_count']    = siteid_cnt[row['siteid']]
            new_row['offerid_count']   = offerid_cnt[row['offerid']]
            new_row['merchant_count']  = merchant_cnt[row['merchant']]
            
        elif feature_type == 'site':
            new_row['siteid_count']    = siteid_cnt[row['siteid']]
        elif feature_type == 'offer':
            new_row['offerid_count']   = offerid_cnt[row['offerid']]
        else:
            new_row['merchant_count']  = merchant_cnt[row['merchant']]
            
            
        new_row['category_count']    = category_cnt[row['category']]
        new_row['hour_count']        = hour_cnt[row['hour']]
        new_row['browser_count']     = browser_cnt[row['browserid']]
        new_row['devid_count']       = devid_cnt[row['devid']]
        new_row['country_count']     = country_cnt[row['countrycode']]
        new_row['site_category_bag'] = len(site_category_bag[row['siteid']])
        
        
        
        writer.writerow(new_row)
        
scan(args['tr_src_path'], args['feature_type'])
scan(args['te_src_path'], args['feature_type'])

print('===================SCAN COMPLETE==================================')
gen_data(args['tr_src_path'], args['tr_dst_path'], args['tr_is_train'] == 't', args['feature_type'])
gen_data(args['te_src_path'], args['te_dst_path'], args['te_is_train'] == 't', args['feature_type'])