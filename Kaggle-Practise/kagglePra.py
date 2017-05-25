import pandas as pd
import numpy as np

#载入数据

train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')

#显示一下数据的大小
#可以通过train.shape, test.shape查看

train.dtypes#查看每个属性的类型
train.head(5)#查看前五条数据
#合并成一个总的data
train['source']='train'
test['source']='test'
data=pd.concat([train, test], ignore_index=True)



#在实际的应用中很重的是观察异常点，比如说异常值;
data.apply(lambda x:sum(x.isnull()))

#要对数据有更深的认识，比如说，咱们看看这些字段，分别有多少种取值(甚至你可以看看分布)
var = ['Gender','Salary_Account','Mobile_Verified','Var1','Filled_Form','Device_Type','Var2','Source']
for v in var:
    print '\n%s这一列数据的不同取值和出现的次数\n'%v
    print data[v].value_counts()

#查看每个字段的个数
data.count()

# 把月、日、和 小时单独拎出来，放到3列中
data['month'] = pd.DatetimeIndex(df_train.datetime).month
data['day'] = pd.DatetimeIndex(df_train.datetime).dayofweek
data['hour'] = pd.DatetimeIndex(df_train.datetime).hour


#简单的赋值操作咯
df_train_origin = df_train
# 抛掉不要的字段，一次删除多个
df_train = df_train.drop(['datetime','casual','registered'], axis = 1)

#看看某一个字段与多少的不同的取值
len(data['City'].unique())#data['City'].unique()给出的结果是在该属性下不同城市的名字都列出来

#如果要删除该属性，可以用下面的额代码
data.drop('City',axis=1,inplace=True)



#DOB是出生的具体日期，咱们要具体日期作用没那么大，年龄段可能对我们有用，所有算一下年龄好了
#创建一个年龄的字段Age
data['Age'] = data['DOB'].apply(lambda x: 115 - int(x[-2:]))

data.drop('DOB',axis=1,inplace=True)#删除原先的字段


data.boxplot(column=['EMI_Loan_Submitted'],return_type='axes')#画出箱线图

#好像缺失值比较多，干脆就开一个新的字段，表明是缺失值还是不是缺失值
data['EMI_Loan_Submitted_Missing'] = data['EMI_Loan_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data[['EMI_Loan_Submitted','EMI_Loan_Submitted_Missing']].head(10)
data.drop('EMI_Loan_Submitted',axis=1,inplace=True)
#看看个数
len(data['Employer_Name'].value_counts())
#看看某一字段的信息
data['Existing_EMI'].describe()
#缺省值不多，用均值代替
data['Existing_EMI'].fillna(0, inplace=True)

#找中位数去填补缺省值（因为缺省的不多）
data['Loan_Amount_Applied'].fillna(data['Loan_Amount_Applied'].median(),inplace=True)
data['Loan_Tenure_Applied'].fillna(data['Loan_Tenure_Applied'].median(),inplace=True)


#处理source
data['Source'] = data['Source'].apply(lambda x: 'others' if x not in ['S122','S133'] else x)
data['Source'].value_counts()

#数值编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['Device_Type','Filled_Form','Gender','Var1','Var2','Mobile_Verified','Source']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])

#类别型的One-Hot 编码
data = pd.get_dummies(data, columns=var_to_encode)
data.columns#看看one-hot编码后的列是哪些
#区分训练和测试数据
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

train.drop('source',axis=1,inplace=True)
#保存代码
train.to_csv('train_modified.csv',index=False)
test.to_csv('test_modified.csv',index=False)

'''
通常遇到缺值的情况，我们会有几种常见的处理方式

    如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了
    如果缺值的样本适中，而该属性非连续值特征属性(比如说类目属性)，那就把NaN作为一个新类别，加到类别特征中
    如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中。
    有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。
    本例中，后两种处理方式应该都是可行的，我们先试试拟合补全吧(虽然说没有特别多的背景可供我们拟合，这不一定是一个多么好的选择)

我们这里用scikit-learn中的RandomForest来拟合一下缺失的年龄数据

'''

from sklearn.ensemble import RandomForestRegressor
 
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])#就是从第一个到最后一个
    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges
    
    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
data_train