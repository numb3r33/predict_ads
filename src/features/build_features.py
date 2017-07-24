"""
This module is responsible for creating features.
"""

import numpy as np

def create_features(train, test):
	# 1. Decompose datetime to hour of the day
	train = train.assign(hour_of_day=train.datetime.dt.hour)
	test  = test.assign(hour_of_day=test.datetime.dt.hour)

	# 2. Check if it is prime time or not
	train = train.assign(prime_time=train.hour_of_day.isin([0, 1, 19, 20, 21, 22, 23]).astype(np.int))
	test  = test.assign(prime_time=test.hour_of_day.isin([0, 1, 19, 20, 21, 22, 23]).astype(np.int))

	# 3. Create feature for siteid count, because it is assumed that sites with same frequency count
	#    behave similarly 

	train = train.assign(siteid_count=train.groupby('siteid')['siteid'].transform(lambda x: len(x)))
	test  = test.assign(siteid_count=test.groupby('siteid')['siteid'].transform(lambda x: len(x)))

	# Since siteid has missing values so siteid_count would have missing values as well, fill those holes
	# with zero
	train.loc[:, 'siteid_count'] = train.siteid_count.fillna(0)
	test.loc[:, 'siteid_count']  = test.siteid_count.fillna(0)


	return train, test