# -*- coding: utf-8 -*-
# @Time    : 2018/12/27 20:55
# @Author  : 朝天椒
# @公众号  : 辣椒哈哈
# @FileName: gbm_xgb
# @Software: PyCharm

from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import random


def get_accuracy(y_predict, y_val):
    result = np.abs(y_predict - y_val) / y_val
    return result


def data_clean(train_paths, val_paths):

    return None

train_path = ''
val_path = ''

gbm_params = {
    "boosting_type": 'gbdt', # default='gbdt'
    "learning_rate": 0.1, # default=0.1, boosting learning rate
    "num_leaves": int(np.power(2, 12)), # default=31 Maximum tree for base learners
    "max_depth": -1, # default=-1,no limit, with the num_leaves have common functions
    "lambda_l2": 0, # default=0, L1 regularization term on weights
    "lambda_l1": 0, # default=0, L2 regularization term on weights
    # prevent overfit parameter as(subsample, subsample_freq, feature_fraction)
    "subsample": 1.0, # default=1, aliases(bagging_fraction) subsample ratio of the training instance
    "subsample_freq": 3, # default=0, 0 means disable bagging; k means perform bagging at every k iteration
    "feature_fraction": 1.0, # default=1.0, choose much feature to train the model

    "objective": 'regression_l1', # default=None, regression for LGBMRegressor ,binary for LGBMClassifier
    "num_iterations": 500, # default=100, use much tree to fit boosting the data

    "colsample_bytree": 0.656, # default=1, subsample ratio of columns when constructing tree
    "random_state": 1000, # default=None, Random number seed
    "n_jobs": 8, # default=-1, Number of parallel threads
    "min_child_weight": 8, # default=1e-3, Minimum sum of instance weight (hessian) needed in a child (leaf)
    "min_child_samples": 10, # default=20, Minimum number of data needed in a child(leaf)
    "min_split_gain": 0.2, # default=0, Minimum loss reduction required to make a further partition on a leaf node of the tree.
    "metric": "l1",
}

xgb_params = {
    'booster': 'gbtree', # default=gbtree, the type of boost tree
    "n_estimators": 100, # default=100, number of boosted trees to fit
	"verbosity": 10, # default=1, verbosity of printing messages
	"n_jobs": 8, # default to maximum number of threads available
	"learning_rate": 0.3, # default=0.3, step size shrinkage used in update
	"min_split_loss": 0, # default=0, minimum loss reduction required to make partition
	"objective": 'reg:logistic', # the task to processing
	'eval_metric': 'mae', # evaluate the loss
    "random_state": 200, # random number seed
	'max_depth':4, # default=6, maximum depth of a tree
	'lambda':10, # default =1, L2 regularization term on weights
    'alpha': 0, # default=0, L1 regularization term on weights
	'subsample':0.75, # default=1, subsample ratio of the training instances
	'colsample_bytree':0.75, # default feature of tree
	'min_child_weight':2, # minimum sum of instance weight needed in a child
}
