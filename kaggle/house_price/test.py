# coding=utf-8
import kaggle.utils.stacking.StackingUtils as su
import kaggle.utils.houseprice.HousePriceUtils as hpu
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

x_data, y_data, x_test_data = hpu.get_data()

y_train = np.asarray(pd.concat([y_data, x_data['Id']], axis=1).values)

x_train = np.asarray(x_data.drop(['Id'], axis=1).values)
x_test = np.asarray(x_test_data.drop(['Id'], axis=1).values)

x_train = Imputer().fit_transform(x_train)
y_train = Imputer().fit_transform(y_train)[:, 1]
print(y_train)
x_test = Imputer().fit_transform(x_test)
et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500
}

rd_params = {
    'alpha': 10
}

ls_params = {
    'alpha': 0.005
}

NFOLDS = 5
SEED = 0

su.stacking(NFOLDS, SEED, et_params, rf_params, xgb_params, rd_params, ls_params, x_train, y_train, x_test)

