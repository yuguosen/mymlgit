# coding=utf-8
# 多个模型之间的融合

import pandas as pd
import numpy as np
from scipy.stats import skew
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt

NTRAIN = 100
NTEST = 0
X_train = []
Y_train = []
X_test = []
NFOLDS = 3
SEED = 0
kf = KFold(NTRAIN, n_folds=NFOLDS, shuffle=True, random_state=SEED)


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf):
    oof_train = np.zeros((NTRAIN,))
    oof_test = np.zeros((NTEST,))
    oof_test_skf = np.empty((NFOLDS, NTEST))

    for i, (train_index, test_index) in enumerate(kf):
        print("TRAIN:", train_index, "TEST:", test_index)

        x_tr = X_train[train_index]
        y_tr = Y_train[train_index]
        x_te = X_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def stacking(nfolds, seed, et_params, rf_params, xgb_params, rd_params, ls_params, x_train, y_train, x_test):
    NTRAIN = x_train.shape[0]
    NTEST = len(x_test)
    NFOLDS = nfolds
    SEED = seed
    X_train = x_train
    Y_train = y_train
    X_test = x_test

    kf = KFold(NTRAIN, n_folds=NFOLDS, shuffle=True, random_state=SEED)

    xg = XgbWrapper(seed=SEED, params=xgb_params)
    et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
    rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
    rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
    ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)

    xg_oof_train, xg_oof_test = get_oof(xg)
    et_oof_train, et_oof_test = get_oof(et)
    rf_oof_train, rf_oof_test = get_oof(rf)
    rd_oof_train, rd_oof_test = get_oof(rd)
    ls_oof_train, ls_oof_test = get_oof(ls)

    print("XG-CV: {}".format(sqrt(mean_squared_error(Y_train, xg_oof_train))))
    print("ET-CV: {}".format(sqrt(mean_squared_error(Y_train, et_oof_train))))
    print("RF-CV: {}".format(sqrt(mean_squared_error(Y_train, rf_oof_train))))
    print("RD-CV: {}".format(sqrt(mean_squared_error(Y_train, rd_oof_train))))
    print("LS-CV: {}".format(sqrt(mean_squared_error(Y_train, ls_oof_train))))

    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test)

    res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=4, seed=SEED, stratified=False,
                 early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 0]
    cv_std = res.iloc[-1, 1]

    print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

    gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
    submission = pd.read_csv("../house_price/data/submission_xgboosting2.csv")
    submission.iloc[:, 1] = gbdt.predict(dtest)
    print(submission['SalePrice'])
    saleprice = np.exp(submission['SalePrice']) - 1
    print(saleprice)
    submission['SalePrice'] = saleprice
    submission.to_csv('xgstacker_starter.sub.csv', index=None)
