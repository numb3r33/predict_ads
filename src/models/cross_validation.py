"""
This module will create training, validation and holdout sets

Options available would be:

1. mask as to how to create training, validation and holdout set
"""

def create_splits(X, y, training_mask, validation_mask, holdout_mask):
	if holdout_mask is None:
		X_train = X.loc[training_mask, :]
		y_train = y.loc[training_mask]

		X_val   = X.loc[validation_mask, :]
		y_val   = y.loc[validation_mask]

		return X_train, X_val, y_train, y_val

	else:
		X_train = X.loc[training_mask, :]
		y_train = y.loc[training_mask]

		X_val   = X.loc[validation_mask, :]
		y_val   = y.loc[validation_mask]

		X_test  = X.loc[holdout_mask, :]
		y_test  = y.loc[holdout_mask]


		return X_train, X_val, X_test, y_train, y_val, y_test