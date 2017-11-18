# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/11/18 20:39'
__site__ = ''
__software__ = 'PyCharm'
__file__ = 'pandas cookbook.py'

import pandas as pd


def read():
    train = pd.read_csv('train_modified.csv')
    test = pd.read_csv('test_modified.csv')
    return train, test


def some():
    train, test = read()
    train['source'] = 'train'
    test['source'] = 'test'
    data = pd.concat([train, test], ignore_index=True)  # 如何区分训练集合和测试集合

    # 查看每个列出现的数据的次数
    for v in data.columns:
        print('\n%s这一列数据的不同取值和出现的次数\n' % v)
        print(data[v].value_counts())

    # 查看某一列的有多少类型的值的个数
    len_of_city = len(data['City'].unique())

    # 删除某一列
    del data['city']
    data.drop('City', axis=1, inplace=True)  # 按照列删除，其中city是列的名称
    data.drop(1, axis=0, inplace=True)  # 按照行删除，其中1是索引的名称

    # 查看离散值
    data.boxplot(column=['EMI_Loan_Submitted'], return_type='axes')  # 做箱线图

    # 中位数填充缺失值
    # 找中位数去填补缺省值（因为缺省的不多）
    data['Loan_Amount_Applied'].fillna(data['Loan_Amount_Applied'].median(), inplace=True)
    data['Loan_Tenure_Applied'].fillna(data['Loan_Tenure_Applied'].median(), inplace=True)

    # 数值编码
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    var_to_encode = ['Device_Type', 'Filled_Form', 'Gender', 'Var1', 'Var2', 'Mobile_Verified', 'Source']
    for col in var_to_encode:
        data[col] = le.fit_transform(data[col])

    # 做one-hot encoding
    data = pd.get_dummies(data, columns=var_to_encode)

    # 时间转换
    # 把月、日、和 小时单独拎出来，放到3列中
    data['month'] = pd.DatetimeIndex(data['datetime']).month
    data['day'] = pd.DatetimeIndex(data['datetime']).dayofweek
    data['hour'] = pd.DatetimeIndex(data['datetime']).hour


from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score
def about_cv():

    # 总得切分一下数据咯（训练集和测试集）
    df_train = pd.read_csv('kaggle_bike_competition_train.csv', header=0)
    df_train = df_train.drop(['datetime', 'casual', 'registered'], axis=1)
    df_train_target = df_train['count'].values
    df_train_data = df_train.drop(['count'], axis=1).values
    cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=3, test_size=0.2,
                                       random_state=0)
    # 各种模型来一圈

    print("岭回归")

    for train, test in cv:
        svc = linear_model.Ridge().fit(df_train_data[train], df_train_target[train])
        print("train score: {0:.3f}, test score: {1:.3f}\n".format(
            svc.score(df_train_data[train], df_train_target[train]),
            svc.score(df_train_data[test], df_train_target[test])))

    print("支持向量回归/SVR(kernel='rbf',C=10,gamma=.001)")

    for train, test in cv:
        svc = svm.SVR(kernel='rbf', C=10, gamma=.001).fit(df_train_data[train], df_train_target[train])
        print("train score: {0:.3f}, test score: {1:.3f}\n".format(
            svc.score(df_train_data[train], df_train_target[train]),
            svc.score(df_train_data[test], df_train_target[test])))

    print("随机森林回归/Random Forest(n_estimators = 100)")

    for train, test in cv:
        svc = RandomForestRegressor(n_estimators=100).fit(df_train_data[train], df_train_target[train])
        print("train score: {0:.3f}, test score: {1:.3f}\n".format(
            svc.score(df_train_data[train], df_train_target[train]),
            svc.score(df_train_data[test], df_train_target[test])))

    X = df_train_data
    y = df_train_target

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.2, random_state=0)

    tuned_parameters = [{'n_estimators': [10, 100, 500]}]

    # 加入了r2 score的参数寻找
    scores = ['r2']

    for score in scores:

        print(score)

        clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)

        print("别！喝！咖！啡！了！最佳参数找到了亲！！：")
        print("")

        # best_estimator_ returns the best estimator chosen by the search
        print(clf.best_estimator_)
        print("得分分别是:")
        # grid_scores_的返回值:
        #    * a dict of parameter settings
        #    * the mean score over the cross-validation folds
        #    * the list of scores for each fold
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

import matplotlib.pyplot as plt
import numpy as np
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    return plt

def dataframe_plot():
    df_train = pd.read_csv('kaggle_bike_competition_train.csv', header=0)
    df_train = df_train.drop(['datetime', 'casual', 'registered'], axis=1)
    df_train_origin = df_train

    # 画图
    # 风速
    df_train_origin.groupby('windspeed').mean().plot(y='count', marker='o')  # 横坐标是风速，纵坐标是该风速下的样本个数
    plt.show()
    # 温度湿度变化
    df_train_origin.plot(x='temp', y='humidity', kind='scatter')  # 很坐标是temp，纵坐标是humidity画图
    plt.show()

    # scatter一下各个维度
    fig, axs = plt.subplots(2, 3, sharey=True)
    df_train_origin.plot(kind='scatter', x='temp', y='count', ax=axs[0, 0], figsize=(16, 8), color='magenta')
    df_train_origin.plot(kind='scatter', x='atemp', y='count', ax=axs[0, 1], color='cyan')
    df_train_origin.plot(kind='scatter', x='humidity', y='count', ax=axs[0, 2], color='red')
    df_train_origin.plot(kind='scatter', x='windspeed', y='count', ax=axs[1, 0], color='yellow')
    df_train_origin.plot(kind='scatter', x='month', y='count', ax=axs[1, 1], color='blue')
    df_train_origin.plot(kind='scatter', x='hour', y='count', ax=axs[1, 2], color='green')


    # 查看相关度
    corr = df_train_origin[['temp', 'weather', 'windspeed', 'day', 'month', 'hour', 'count']].corr()
    print(corr)
    # 用颜色深浅来表示相关度
    plt.figure()
    plt.matshow(corr)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # ================查看学习曲线============================================
    df_train = pd.read_csv('kaggle_bike_competition_train.csv', header=0)
    df_train = df_train.drop(['datetime', 'casual', 'registered'], axis=1)
    df_train_target = df_train['count'].values
    df_train_data = df_train.drop(['count'], axis=1).values
    df_train_origin = df_train

    X = df_train_data
    y = df_train_target

    title = "Learning Curves (Random Forest, n_estimators = 100)"
    cv = cross_validation.ShuffleSplit(df_train_data.shape[0], n_iter=10, test_size=0.2, random_state=0)
    estimator = RandomForestRegressor(n_estimators=100)
    plot_learning_curve(estimator, title, X, y, (0.0, 1.01), cv=cv, n_jobs=4)

    plt.show()
    # ================查看学习曲线============================================
    # 看你们自己的咯


