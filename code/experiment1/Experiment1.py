
# coding: utf-8

# In[51]:

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import scipy as sp
import time
import gc

from csv import DictReader

import matplotlib.pyplot as plt
import seaborn as sns

SEED = 972
np.random.seed(SEED)

import warnings
warnings.filterwarnings('ignore')


# ** Load Training data **

# In[38]:

dtypes = {
    'siteid' : 'float32',
    'offerid': 'uint32',
    'category': 'uint32',
    'merchant': 'uint32'
}


# In[44]:

train = pd.read_csv('../data/raw/205e1808-6-dataset/train.csv', 
                   dtype=dtypes, 
                   parse_dates=['datetime'])


# ** Follow the regularized leader algorithm (FTRL) **

# In[58]:

class ftrl(object):
    
    def __init__(self, alpha, beta, l1, l2, bits):
        self.z = [0.] * bits
        self.n = [0.] * bits
        self.alpha = alpha
        self.beta  = beta
        self.l1    = l1
        self.l2    = l2
        self.w     = {}
        self.X     = []
        self.y     = 0.
        self.bits  = bits
        self.Prediction = 0.
        
    def sgn(self, x):
        return -1 if x < 0 else 1
    
    def fit(self, line):
        try:
            self.ID = line['ID']
            del line['ID']
        except:
            pass
        
        try:
            self.y = float(line['click'])
            del line['click']
        except:
            pass
        
        del line['datetime'], line['siteid'], line['devid']
        
        self.X = [0.] * len(line)
        
        for i, key in enumerate(line):
            val = line[key]
            self.X[i] = (np.abs(hash(key + '_' + val)) % self.bits)
        
        self.X = [0] + self.X
        
    def logloss(self):
        act  = self.y
        pred = self.Prediction
        predicted = np.max(np.min(pred, 1 - 10e-15), 10e-15)
        return -np.log(predicted) if act == 1. else -np.log(1. - predicted)
    
    def predict(self):
        W_dot_x = 0.
        w = {}
        
        for i in self.X:
            if np.abs(self.z[i]) <= self.l1:
                w[i] = 0.
            else:
                w[i] = (self.sgn(self.z[i]) * self.l1 - self.z[i]) / (((self.beta + np.sqrt(self.n[i]))/self.alpha) + self.l2)
        W_dot_x += w[i]
        self.w = w
        self.Prediction = 1. / (1. + np.exp(-max(min(W_dot_x, 35.), -35.)))
        return self.Prediction
    
    def update(self, prediction): 
        for i in self.X:
            g = (prediction - self.y)
            sigma = (1./self.alpha) * (np.sqrt(self.n[i] + g*g) - np.sqrt(self.n[i]))
            self.z[i] += g - sigma*self.w[i]
            self.n[i] += g*g


# In[65]:

clf = ftrl(alpha = 0.1, 
           beta = 1., 
           l1 = 0.1,
           l2 = 1.0, 
           bits = 20)

loss = 0.
count = 0

for t, line in enumerate(DictReader(open('../data/raw/205e1808-6-dataset/train.csv'), delimiter=',')):
    clf.fit(line)
    pred = clf.predict()
    loss += clf.logloss()
    clf.update(pred)
    count += 1
    if count%10000 == 0: 
        print ("(seen, loss) : ", (count, loss * 1./count))
    if count == 100000: 
        break


# In[66]:

test = '../data/raw/205e1808-6-dataset/test.csv'
with open('../data/interim/temp.csv', 'w') as output:
    for t, line in enumerate(DictReader(open(test), delimiter=',')):
        clf.fit(line)
        output.write('%s\n' % str(clf.predict()))
    
    output.close()


# In[76]:

# load submission
sub   = pd.read_csv('../data/raw/205e1808-6-dataset/sample_submission.csv')
probs = pd.read_csv('../data/interim/temp.csv', header=None) 


# In[78]:

sub.loc[:, 'click'] = probs[0]


# In[82]:

sub.to_csv('../submissions/baseline_submission.csv.gz', index=False, compression='gzip')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



