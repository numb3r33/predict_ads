"""
This module will load data from disk

Provides options as:

1. Load data as it is without explicitly mentioning data types
2. Whether to load both train and test dataset or not
"""

import pandas as pd


def load(train_path, test_path, include_dtypes):
	if include_dtypes:
		
		dtypes = {
			'siteid': 'float32',
			'offerid': 'uint32',
			'category': 'uint32',
			'merchant': 'uint32'
		}

	if train_path:
		if include_dtypes:
			train = pd.read_csv(train_path, dtype=dtypes, parse_dates=['datetime'])
		else:
			train = pd.read_csv(train_path, parse_dates=['datetime'])
	
	if test_path:
		if include_dtypes:
			test = pd.read_csv(test_path, dtype=dtypes, parse_dates=['datetime'])
		else:
			test = pd.read_csv(test_path, dtype=dtypes, parse_dates=['datetime'])

	return train, test

