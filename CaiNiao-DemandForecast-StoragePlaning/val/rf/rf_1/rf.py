#coding=utf-8
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os,cPickle,random


#计算代价
def cal_cost(y,pred,a,b):
    cost = 0
    nb_sample = len(y)
    for i in range(nb_sample):
        if pred[i]>y[i]:
            cost += b[i]*(pred[i]-y[i])
        else:
            cost += a[i]*(y[i]-pred[i])
    return nb_sample,cost


a_b = pd.read_csv('../../../data/config1.csv')


data = pd.read_csv('../../../data/train_test.csv')
hashmap = {1:'1',2:'2',3:'3',4:'4',5:'5',6:'all'}
data.store_code = data.store_code.apply(lambda x:hashmap[x])

data = pd.merge(data,a_b,on=['item_id','store_code'])


data = data[data.store_code=='1']






def pipeline():
        val = data[data.watch==1]
        val_a_b = val[['item_id','store_code','a','b']]
        val_y = val.label
        val_x = val.drop(['label','watch','item_id','store_code','a','b'],axis=1)

        train = data[(data.watch!=1)&(data.watch!=0)]
        train_y = train.label

        
        a = list(train.a)
        b = list(train.b)
        train_weight = []
        for i in range(len(a)):
            train_weight.append(min(a[i],b[i]))
        train_weight = np.array(train_weight)

        train_x = train.drop(['label','watch','item_id','store_code','a','b'],axis=1)

        train_x.fillna(train_x.median(),inplace=True)
        val_x.fillna(val_x.median(),inplace=True)
        

        model = RandomForestRegressor(n_estimators=500,max_depth=7,max_features=0.8,n_jobs=-1,random_state=1024)

	#train
	model.fit(train_x,train_y, sample_weight=train_weight)


	#predict val set
	val_a_b['pred'] = model.predict(val_x)
	val_a_b['y'] = val_y
	cost = cal_cost(val_y.values,val_a_b.pred.values,val_a_b.a.values,val_a_b.b.values)
        val_a_b.to_csv('val_{0}.csv'.format(cost[1]),index=None)


if __name__ == "__main__":
    pipeline()
