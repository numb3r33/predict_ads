import argparse, csv, collections, time
import numpy as np

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path', type=str)
parser.add_argument('te_src_path', type=str)
parser.add_argument('tr_dst_path', type=str)
parser.add_argument('te_dst_path', type=str)
parser.add_argument('tr_is_train', type=str)
parser.add_argument('te_is_train', type=str)
parser.add_argument('feature_type',type=str)
parser.add_argument('tr_merge_leaves',type=str)
parser.add_argument('te_merge_leaves',type=str)
parser.add_argument('leaves0_src_path',type=str, default='')
parser.add_argument('leaves1_src_path', type=str, default='')

args = vars(parser.parse_args())

if args['feature_type'] == 'all':
    fields = ['hour', 'browserid', 'devid', 'countrycode',
              'siteid', 'offerid', 'merchant', 'category',
              'siteid_count', 'offerid_count',
              'merchant_count', 'category_count',
              'browser_count',
              'devid_count', 'country_count',
              'hour_count', 'site_category_bag'
             ]

elif args['feature_type'] == 'site':
    fields = ['hour', 'browserid', 'devid', 'countrycode',
              'siteid', 'category', 'merchant',
              'siteid_count', 'offerid_count',
              'merchant_count',
              'browser_count', 'devid_count',
              'country_count', 'hour_count', 'site_category_bag'
             ]
    
elif args['feature_type'] == 'offer':
    fields = ['hour', 'browserid', 'devid', 'countrycode',
              'offerid', 'category', 'merchant',
              'offerid_count', 'merchant_count', 
              'browser_count', 'devid_count',
              'country_count', 'hour_count', 'site_category_bag'
             ]
    
else:
    fields = ['hour', 'browserid', 'devid', 'countrycode',
              'category', 'merchant', 'merchant_count',
              'browser_count', 'devid_count',
              'country_count', 'hour_count', 'site_category_bag'
             ]

start = time.clock()

def convert(src_path, dst_path, is_train, merge_leaves, leaves_src_path):
    with open(dst_path, 'w') as f:
        
        if merge_leaves:
            leaves  = np.load(leaves_src_path)
                       
        for i, row in enumerate(csv.DictReader(open(src_path)), start=1):
            if i % 1000000 == 0:
                print('Took: {} seconds'.format((time.clock() - start)))
            
            feats = []
            
            for j, field in enumerate(fields, start=1):
                feats.append(str(j - 1) + ':' + hashstr(row[field]) + ':1')
                
            if merge_leaves:
                    
                lf      = []
                lf_name = 'leaf'

                for k, leaf in enumerate(leaves[i-1], start=1):
                    lf.append(lf_name + str(k)+':' + hashstr(str(leaf)) + ':1')
                    
            if is_train:
                if merge_leaves:
                    f.write('{0} {1} {2}\n'.format(row['click'], ' '.join(feats), ' '.join(lf)))
                else:
                    f.write('{0} {1}\n'.format(row['click'], ' '.join(feats)))
                    
            else:
                if merge_leaves:
                    f.write('1 {0} {1}\n'.format(' '.join(feats), ' '.join(lf)))
                else:
                    f.write('1 {0}\n'.format(' '.join(feats)))

                    

convert(args['tr_src_path'], args['tr_dst_path'], args['tr_is_train'] == 't', 
        args['tr_merge_leaves'] == 'y', args['leaves0_src_path'])
convert(args['te_src_path'], args['te_dst_path'], args['te_is_train'] == 't',
        args['te_merge_leaves'] == 'y', args['leaves1_src_path'])
                