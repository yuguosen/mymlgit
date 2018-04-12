# coding=utf-8
# xgboost参数调优过程
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# 训练出最合适的n_estimators值
def train_n_estimators(x_train, y_train, times, learning_rate):
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:linear',
        nthread=4,
        scale_pos_weight=1,
        n_estimators=40,
        seed=27
    )

    n_estimators = [50, 70, 100, 200, 300, 400, 600, 800, 1000]

    best_n_estimators = 0
    best_n_estimators_scors = -float('Inf')
    for time_index in range(0, times):
        param_grid = dict(n_estimators=n_estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
        grid_result = grid_search.fit(x_train, y_train)
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        b_score = grid_result.best_score_
        with open('../house_price/data/n_estimators.txt', 'a')as f:
            for mean, stdev, param in zip(means, stds, params):
                f.write("%f (%f) with: %r \n" % (mean, stdev, param))
            f.write("best_params:%s ,best_score:%f \n" % (grid_result.best_params_, grid_result.best_score_))
        best_n_estimators = grid_result.best_params_['n_estimators']
        print(best_n_estimators)
        if b_score >= best_n_estimators_scors:
            best_n_estimators_index = n_estimators.index(best_n_estimators)
            print("best_n_estimators_scors:%f,best_n_estimators:%f,best_n_estimators_index:%f" % (
                best_n_estimators_scors, best_n_estimators, best_n_estimators_index))
            best_n_estimators_scors = b_score
            first_n = 0
            last_n = 0
            if best_n_estimators_index == 0:
                last_n = round(
                    best_n_estimators + (n_estimators[best_n_estimators_index + 1] - best_n_estimators) / 2)
                n_estimators = [best_n_estimators, last_n]
            elif best_n_estimators_index == (len(n_estimators) - 1):
                interval = (best_n_estimators - n_estimators[best_n_estimators_index - 1]) / 2
                first_n = round(
                    best_n_estimators - (best_n_estimators - n_estimators[best_n_estimators_index - 1]) / 2)
                n_estimators = [first_n, best_n_estimators]
            else:
                first_n = round(
                    best_n_estimators - (n_estimators[best_n_estimators_index - 1] - best_n_estimators) / 2)
                last_n = round(
                    best_n_estimators + (n_estimators[best_n_estimators_index + 1] - best_n_estimators) / 2)
                n_estimators = [first_n, best_n_estimators, last_n]

    return best_n_estimators


# 在n_estimators确定的情况下，max_depth，min_weight参数调优
def train_max_depth_min_weight(x_train, y_train, learning_rate, n_estimators):
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        gamma=0,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:linear',
        nthread=4,
        scale_pos_weight=1,
        n_estimators=n_estimators,
        seed=27
    )
    param_test = {
        'max_depth': range(3, 10, 1),
        'min_child_weight': range(1, 6, 1)
    }
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_test, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    with open('../house_price/data/max_depth_min_weight.txt', 'a')as f:
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r \n" % (mean, stdev, param))
        f.write("best_params:%s ,best_score:%f \n" % (grid_result.best_params_, grid_result.best_score_))

    return grid_result.best_params_['max_depth'], grid_result.best_params_['min_child_weight']


# 在n_estimators,max_depth，min_weight确定的情况下,gamma参数调优
def train_gamma(x_train, y_train, learning_rate, n_estimators, max_depth, min_child_weight):
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:linear',
        nthread=4,
        scale_pos_weight=1,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        seed=27
    )
    param_gamma = {
        'gamma': [i / 10.0 for i in range(0, 20)]
    }
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_gamma, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    with open('../house_price/data/gamma.txt', 'a')as f:
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r \n" % (mean, stdev, param))
        f.write("best_params:%s ,best_score:%f \n" % (grid_result.best_params_, grid_result.best_score_))

    return grid_result.best_params_['gamma']


def train_subsample_colsample_bytree(x_train, y_train, learning_rate, n_estimators, max_depth, min_child_weight, gamma):
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        gamma=gamma,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:linear',
        nthread=4,
        scale_pos_weight=1,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        seed=27
    )
    param_test = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_test, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    with open('../house_price/data/subsample_colsample_bytree.txt', 'a')as f:
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r \n" % (mean, stdev, param))
        f.write("best_params:%s ,best_score:%f \n" % (grid_result.best_params_, grid_result.best_score_))

    return grid_result.best_params_['subsample'], grid_result.best_params_['colsample_bytree']


def train_learning_rate(x_train, y_train, learning_rate, n_estimators, max_depth, min_child_weight, gamma, subsample,
                        colsample_bytree):
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective='reg:linear',
        nthread=4,
        scale_pos_weight=1,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        seed=27
    )
    params_learning_rate = {
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.14, 0.16, 0.18, 0.20, 0.22]
    }
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, params_learning_rate, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    with open('../house_price/data/learning_rate.txt', 'a')as f:
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r \n" % (mean, stdev, param))
        f.write("best_params:%s ,best_score:%f \n" % (grid_result.best_params_, grid_result.best_score_))

    return grid_result.best_params_['learning_rate']


def train_all(x_train, y_train):
    model = xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:linear',
        nthread=4,
        scale_pos_weight=1,
        n_estimators=40,
        seed=27
    )
    # 设置参数
    parameters = [
        {
            "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.08, 0.09, 0.11, 0.13, 0.14],
            'n_estimators': [10, 100, 200, 300, 400, 500, 600, 700, 800, 800],
            'max_depth': range(3, 10, 1),
            'min_child_weight': range(1, 6, 1),
            'gamma': [i / 10.0 for i in range(0, 10)],
            'subsample': [i / 10.0 for i in range(2, 10)],
            'colsample_bytree': [i / 10.0 for i in range(2, 10)]
        }
    ]
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, parameters, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    with open('../house_price/data/train_all.txt', 'a')as f:
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r \n" % (mean, stdev, param))
        f.write("best_params:%s ,best_score:%f \n" % (grid_result.best_params_, grid_result.best_score_))
