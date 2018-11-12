# coding:utf-8
'''
@time:    Created on  2018-11-11 19:48:37
@author:  Lanqing
@Func:    contest.adaboost
'''

train_fid = 'C:/Users/jhh/Desktop/contest/input/train.csv'
test_fid = 'C:/Users/jhh/Desktop/contest/input/test.csv'

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import xgboost as xgb

train_data = pd.read_csv(train_fid)
test_data = pd.read_csv(test_fid)

df = train_data
# 重采样
df_majority = df[df['TARGET'] == 0]
# df_majority.label = 0
df_minority = df[df['TARGET'] == 1]
from sklearn.utils import resample
df_minority_upsampled = resample(df_minority,
                                        replace=True,  # sample with replacement
                                        n_samples=len(df_majority),  # to match majority class
                                        random_state=123)  # reproducible results
# 合并多数类别同上采样过的少数类别
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
print(df_upsampled['TARGET'].value_counts())
print(df_upsampled.describe())
train_datas1 = df_upsampled
train_datas1 = train_data

train_datas1 = train_datas1.sample(frac=1)


# fetch
train_label = train_datas1['TARGET'].values
train_datas = train_datas1.iloc[:, 0:-1]
test_datas1 = test_data.values
test_datas = test_datas1[:, 0:]

# maxmin
model = MinMaxScaler()
model = model.fit(train_datas)
train_datas = model.transform(train_datas)
# X_train, X_test, y_train, y_test = train_test_split(train_datas, train_label, test_size=0.6, random_state=1, shuffle=True) 

# baseline
model2 = RandomForestClassifier(n_estimators=20)
from sklearn.neighbors import KNeighborsClassifier
# model2 = KNeighborsClassifier()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree

# param = {'bst:max_depth':20, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
# data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
# bst = xgb.train(dtrain=data_dmatrix, params=param)

# model2 = tree.DecisionTreeClassifier()
model2 = RandomForestClassifier(n_estimators=20, max_features=15)
# model2 = KNeighborsClassifier(n_neighbors=5)

X, y = train_datas, train_label
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    # print(y_pred, '\n', y_test)
    # print(X_test.shape)
    s1 = metrics.roc_auc_score(y_test, y_pred)
    f2 = metrics.confusion_matrix(y_test, y_pred)
    f3 = metrics.f1_score(y_test, y_pred, average='macro')
    print('Auc', s1)  # \nConfusionMatrix\n', f2, '\nF1Score', f3) 

import numpy as np
test_datas11 = model.transform(test_datas)
print(test_datas11.shape)
t = model2.predict_proba(test_datas11)
print(t)
t = np.array(t)
t = t[:, 1]  # np.max(t, axis=1)
ID = test_datas1[:, 0]
print(t.shape, ID.shape)
saved = np.hstack((ID.reshape(-1, 1), t.reshape(-1, 1)))
print(saved.shape)
np.savetxt('a.csv', saved, fmt='%d,%.2f')
