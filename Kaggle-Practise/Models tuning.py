from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score
import numpy as np


# �ܵ��з�һ�����ݿ���ѵ�����Ͳ��Լ���
cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=3, test_size=0.2,
    random_state=0)

# ����ģ����һȦ

print "��ع�"    
for train, test in cv:    
    svc = linear_model.Ridge().fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))
    
print "֧�������ع�/SVR(kernel='rbf',C=10,gamma=.001)"
for train, test in cv:
    
    svc = svm.SVR(kernel ='rbf', C = 10, gamma = .001).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))
    
print "���ɭ�ֻع�/Random Forest(n_estimators = 100)"    
for train, test in cv:    
    svc = RandomForestRegressor(n_estimators = 100).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))
#�������Ǹ��������û�������Ҳ�п�������Ĳ������õĲ��԰����������������Լ�ȥ���Կ�����������ң�����
#�ö�ͬѧ�ʲ���զ������������һ�����߿��԰�æ������GridSearch����������ȿ��ȵ�ʱ�򣬰�����ש�����Ҳ���
X = df_train_data
y = df_train_target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.2, random_state=0)

tuned_parameters = [{'n_estimators':[10,100,500]}]   
    
scores = ['r2']

for score in scores:
    
    print score
    
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("�𣡺ȣ������ȣ��ˣ���Ѳ����ҵ����ף�����")
    print ""
    #best_estimator_ returns the best estimator chosen by the search
    print(clf.best_estimator_)
    print ""
    print("�÷ֱַ���:")
    print ""
    #grid_scores_�ķ���ֵ:
    #    * a dict of parameter settings
    #    * the mean score over the cross-validation folds 
    #    * the list of scores for each fold
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print ""

'''
�㿴������Grid Search��������������������ģ���Ҳ���Դ󵨷��ĵ��ڸղ�������ģ������һ�ѡ�
����Ҫ����ģ��״̬�ǲ��ǣ������orǷ���,������ѧϰ����
'''
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


title = "Learning Curves (Random Forest, n_estimators = 100)"
cv = cross_validation.ShuffleSplit(df_train_data.shape[0], n_iter=10,test_size=0.2, random_state=0)
estimator = RandomForestRegressor(n_estimators = 100)
plot_learning_curve(estimator, title, X, y, (0.0, 1.01), cv=cv, n_jobs=4)

plt.show()
'''
�������˰ɣ�ѵ�����Ͳ��Լ�ֱ�Ӽ����ô���⡣����һ���ǹ������
���ɭ�������㷨ѧϰ�����ǳ�ǿ������Ҵ�������Աȸ���ģ�͵÷ֵ�ʱ��Ҳ���Կ�����ѵ�����Ͳ��Լ��ĵ÷�Ҳ�ǲ�����ģ�����ϻ������ԡ�
���ԣ�����˵ʲô�أ������˺˵�ȥ����ȳ棬Ȼ������Ҳ�е㳤����ֲ����
so, �����������ô�����ţ������ش��£����ˣ��ǻ���ȥ����ppt
'''
# ����һ�»������ϣ���Ȼ��δ�سɹ�
print "���ɭ�ֻع�/Random Forest(n_estimators=200, max_features=0.6, max_depth=15)"
svc = RandomForestRegressor(n_estimators = 200, max_features=0.6, max_depth=15).fit(df_train_data[train], df_train_target[train])
print("train score: {0:.3f}, test score: {1:.3f}\n".format(
svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))


#����һЩ������ͳ����
# ����ͳ�ƣ��Է��׷��飬Ȼ�������ֵ---����ͼ
df_train_origin.groupby('windspeed').mean().plot(y='count', marker='o')
plt.show()

#�¶�ʪ�ȱ仯---ɢ��ͼ
df_train_origin.plot(x='temp', y='humidity', kind='scatter')
plt.show()

#�����Կ�����������֮��Ĺ�����
# ��������ضȿ�
corr = df_train_origin[['temp','weather','windspeed','day', 'month', 'hour','count']].corr()
# ����ɫ��ǳ����ʾ��ض�
plt.figure()
plt.matshow(corr)
plt.colorbar()
plt.show()