# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/11/16 22:44'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'XGBoost models tuning.py'

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4
target='Disbursed'
IDcol = 'ID'

def read():
    train = pd.read_csv('train_modified.csv')
    test = pd.read_csv('test_modified.csv')
    return train, test


# test_results = pd.read_csv('test_results.csv')
def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # 建模
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # 对训练集预测
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # 输出模型的一些结果
    print("准确率 : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))

    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))


    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

if __name__ == '__main__':
    train, test = read()
    # param_test1 = {
    #     'max_depth': range(3, 5, 2),
    #     'min_child_weight': range(1, 6, 3),
    #
    #     'gamma': [i / 10.0 for i in range(0, 5)],
    #
    #
    #     'subsample': [i / 100.0 for i in range(75, 90, 5)],
    #     'colsample_bytree': [i / 100.0 for i in range(75, 90, 5)],
    #
    #
    #     'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    # }
    # gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
    #                                                 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #                                                 objective='binary:logistic', nthread=4, scale_pos_weight=1,
    #                                                 seed=27),
    #                         param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    # gsearch1.fit(train[predictors], train[target])
    # for line in gsearch1.grid_scores_:
    #     print(line)
    # print(gsearch1.best_params_)
    # print(gsearch1.best_score_)

    predictors = [x for x in train.columns if x not in [target, IDcol]]
    xgb2 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb2, train, test, predictors)