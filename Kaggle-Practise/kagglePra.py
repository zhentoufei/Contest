import pandas as pd
import numpy as np

#��������

train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')

#��ʾһ�����ݵĴ�С
#����ͨ��train.shape, test.shape�鿴

train.dtypes#�鿴ÿ�����Ե�����
train.head(5)#�鿴ǰ��������
#�ϲ���һ���ܵ�data
train['source']='train'
test['source']='test'
data=pd.concat([train, test], ignore_index=True)



#��ʵ�ʵ�Ӧ���к��ص��ǹ۲��쳣�㣬����˵�쳣ֵ;
data.apply(lambda x:sum(x.isnull()))

#Ҫ�������и������ʶ������˵�����ǿ�����Щ�ֶΣ��ֱ��ж�����ȡֵ(��������Կ����ֲ�)
var = ['Gender','Salary_Account','Mobile_Verified','Var1','Filled_Form','Device_Type','Var2','Source']
for v in var:
    print '\n%s��һ�����ݵĲ�ͬȡֵ�ͳ��ֵĴ���\n'%v
    print data[v].value_counts()

#�鿴ÿ���ֶεĸ���
data.count()

# ���¡��ա��� Сʱ������������ŵ�3����
data['month'] = pd.DatetimeIndex(df_train.datetime).month
data['day'] = pd.DatetimeIndex(df_train.datetime).dayofweek
data['hour'] = pd.DatetimeIndex(df_train.datetime).hour


#�򵥵ĸ�ֵ������
df_train_origin = df_train
# �׵���Ҫ���ֶΣ�һ��ɾ�����
df_train = df_train.drop(['datetime','casual','registered'], axis = 1)

#����ĳһ���ֶ�����ٵĲ�ͬ��ȡֵ
len(data['City'].unique())#data['City'].unique()�����Ľ�����ڸ������²�ͬ���е����ֶ��г���

#���Ҫɾ�������ԣ�����������Ķ����
data.drop('City',axis=1,inplace=True)



#DOB�ǳ����ľ������ڣ�����Ҫ������������û��ô������ο��ܶ��������ã�������һ���������
#����һ��������ֶ�Age
data['Age'] = data['DOB'].apply(lambda x: 115 - int(x[-2:]))

data.drop('DOB',axis=1,inplace=True)#ɾ��ԭ�ȵ��ֶ�


data.boxplot(column=['EMI_Loan_Submitted'],return_type='axes')#��������ͼ

#����ȱʧֵ�Ƚ϶࣬�ɴ�Ϳ�һ���µ��ֶΣ�������ȱʧֵ���ǲ���ȱʧֵ
data['EMI_Loan_Submitted_Missing'] = data['EMI_Loan_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data[['EMI_Loan_Submitted','EMI_Loan_Submitted_Missing']].head(10)
data.drop('EMI_Loan_Submitted',axis=1,inplace=True)
#��������
len(data['Employer_Name'].value_counts())
#����ĳһ�ֶε���Ϣ
data['Existing_EMI'].describe()
#ȱʡֵ���࣬�þ�ֵ����
data['Existing_EMI'].fillna(0, inplace=True)

#����λ��ȥ�ȱʡֵ����Ϊȱʡ�Ĳ��ࣩ
data['Loan_Amount_Applied'].fillna(data['Loan_Amount_Applied'].median(),inplace=True)
data['Loan_Tenure_Applied'].fillna(data['Loan_Tenure_Applied'].median(),inplace=True)


#����source
data['Source'] = data['Source'].apply(lambda x: 'others' if x not in ['S122','S133'] else x)
data['Source'].value_counts()

#��ֵ����
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['Device_Type','Filled_Form','Gender','Var1','Var2','Mobile_Verified','Source']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])

#����͵�One-Hot ����
data = pd.get_dummies(data, columns=var_to_encode)
data.columns#����one-hot������������Щ
#����ѵ���Ͳ�������
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

train.drop('source',axis=1,inplace=True)
#�������
train.to_csv('train_modified.csv',index=False)
test.to_csv('test_modified.csv',index=False)

'''
ͨ������ȱֵ����������ǻ��м��ֳ����Ĵ���ʽ

    ���ȱֵ������ռ�����������ߣ����ǿ��ܾ�ֱ�������ˣ���Ϊ��������Ļ������ܷ�������noise��Ӱ�����Ľ����
    ���ȱֵ���������У��������Է�����ֵ��������(����˵��Ŀ����)���ǾͰ�NaN��Ϊһ������𣬼ӵ����������
    ���ȱֵ���������У���������Ϊ����ֵ�������ԣ���ʱ�����ǻῼ�Ǹ���һ��step(���������age�����ǿ��Կ���ÿ��2/3��Ϊһ������)��Ȼ�������ɢ����֮���NaN��Ϊһ��type�ӵ�������Ŀ�С�
    ��Щ����£�ȱʧ��ֵ�����������ر�࣬������Ҳ�������Ÿ������е�ֵ�����һ�����ݣ������ϡ�
    �����У������ִ���ʽӦ�ö��ǿ��еģ�������������ϲ�ȫ��(��Ȼ˵û���ر��ı����ɹ�������ϣ��ⲻһ����һ����ô�õ�ѡ��)

����������scikit-learn�е�RandomForest�����һ��ȱʧ����������

'''

from sklearn.ensemble import RandomForestRegressor
 
### ʹ�� RandomForestClassifier �ȱʧ����������
def set_missing_ages(df):
    
    # �����е���ֵ������ȡ��������Random Forest Regressor��
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # �˿ͷֳ���֪�����δ֪����������
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y��Ŀ������
    y = known_age[:, 0]

    # X����������ֵ
    X = known_age[:, 1:]

    # fit��RandomForestRegressor֮��
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # �õõ���ģ�ͽ���δ֪������Ԥ��
    predictedAges = rfr.predict(unknown_age[:, 1::])#���Ǵӵ�һ�������һ��
    
    # �õõ���Ԥ�����ԭȱʧ����
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges
    
    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
data_train