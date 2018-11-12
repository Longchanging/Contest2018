# coding:utf-8
'''
@time:    Created on  2018-11-11 17:41:26
@author:  Lanqing
@Func:    contest.main
'''
train_fid = 'C:/Users/jhh/Desktop/contest/input/train.csv'
test_fid = 'C:/Users/jhh/Desktop/contest/input/test.csv'

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

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
# train_datas1 = train_data

# fetch
train_label = train_datas1['TARGET'].values
train_datas = train_datas1.iloc[:, 0:-1]
test_datas1 = test_data.values
test_datas = test_datas1[:, 0:]

# maxmin
model = MinMaxScaler()
model = model.fit(train_datas)
train_datas = model.transform(train_datas)
X_train, X_test, y_train, y_test = train_test_split(train_datas, train_label, test_size=0.01, random_state=1, shuffle=True) 

# baseline
model2 = RandomForestClassifier(n_estimators=100)
from sklearn.neighbors import KNeighborsClassifier
# model2 = KNeighborsClassifier()


model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
print(X_test.shape)
s1 = metrics.roc_auc_score(y_test, y_pred)
f2 = metrics.confusion_matrix(y_test, y_pred)
f3 = metrics.f1_score(y_test, y_pred, average='macro')
print('\nAuc', s1, '\nConfusionMatrix\n', f2, '\nF1Score', f3) 

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
np.savetxt('a.csv', saved, fmt='%d,%.1f')