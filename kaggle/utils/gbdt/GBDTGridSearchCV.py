# coding=utf-8
# gbdt参数调优过程
from sklearn.ensemble import GradientBoostingRegressor as gbdr
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# 训练出最合适的n_estimators值
def train_n_estimators(x_train, y_train, times, learning_rate):
    model = gbdr(
        learning_rate=learning_rate,
        min_samples_split=300,
        min_samples_leaf=20,
        max_depth=8,
        max_features='sqrt',
        subsample=0.8,
        random_state=10
    )

    n_estimators = [10, 20, 30, 40, 50, 70, 100, 200, 300, 400]

    best_n_estimators = 0
    best_n_estimators_scors = -float('Inf')
    for time_index in range(0, times):
        param_grid = dict(n_estimators=n_estimators)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_result = grid_search.fit(x_train, y_train)
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        b_score = grid_result.best_score_
        with open('../house_price/data/gbdt/n_estimators.txt', 'a')as f:
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
