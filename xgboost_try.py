# coding:utf-8
'''
@time:    Created on  2018-11-11 20:46:29
@author:  Lanqing
@Func:    contest.xgboost
'''
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
from random import seed

seed(0)

###########################################################
import pandas as pd
train_fid = 'C:/Users/jhh/Desktop/contest/input/train.csv'
test_fid = 'C:/Users/jhh/Desktop/contest/input/test.csv'
train_data = pd.read_csv(train_fid)
test_data = pd.read_csv(test_fid)
##### Remove columns
remove_names = ['PersonalField39', 'PersonalField64', 'PersonalField69', 'GeographicField10B']
train_data = train_data.drop(columns=remove_names)
test_data = test_data.drop(columns=remove_names)
categori = ['SalesField13', 'PersonalField9', 'PersonalField13', 'PersonalField29',
            'PersonalField34', 'PersonalField44', 'PersonalField49', 'PersonalField54',
            'PersonalField59', 'PersonalField74', 'PropertyField8', 'PropertyField17',
            'GeographicField18A', 'GeographicField23A']
tt1, tt2 = pd.DataFrame(), pd.DataFrame()

# for column in categori:
    # df_train = pd.get_dummies(train_data[column], prefix=column)
    # df_test = pd.get_dummies(test_data[column], prefix=column)
    # tt1 = pd.concat([tt1, df_train], axis=1)
    # tt2 = pd.concat([tt2, df_test], axis=1)
# from keras.utils import to_categorical
# A = pd.get_dummies(train_data[categori], prefix='1')
# B = pd.get_dummies(test_data[categori], prefix='2')
# A, B = A.values + 1, B.values + 1
# print(A.min(), B.min())
# OE = OneHotEncoder()   
# superA = np.vstack((A, B)) 
# OE = OE.fit(superA)
# A = to_categorical(A)
# B = to_categorical(B)
# A, B = np.array(A), np.array(B)
# print(A[0], '\n', B[0])
# tt1 = pd.DataFrame(A)
# tt2 = pd.DataFrame(B)
# 
# train_data = train_data.drop(columns=categori)
# test_data = test_data.drop(columns=categori)
# train_data = pd.concat([train_data, tt1], axis=1)
# test_data = pd.concat([test_data, tt2], axis=1)
# # print(train_data.describe())
# print(train_data.shape, test_data.shape)

train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
print(train_data.shape, test_data.shape)

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
train_datas1 = df_upsampled
train_datas1 = train_datas1.sample(frac=1)

# fetch
train_label = train_datas1['TARGET'].values
train_datas = train_datas1.iloc[:, 1:-1]
test_datas1 = test_data
test_datas = test_datas1.iloc[:, 1:]
# maxmin
model = MinMaxScaler()
model = model.fit(train_datas)
train_datas = model.transform(train_datas)
model = MinMaxScaler()
test_datas11 = model.fit_transform(test_datas)
# PCA
# mod = PCA(n_components=20)
# mod = mod.fit(train_datas)
# train_datas = mod.transform(train_datas)
# test_datas11 = mod.transform(test_datas11)

param = {'n_estimators':100,
         'bst:max_depth':3, 'bst:eta':0.01, 'silent':1,
         # 'num_feature':10,
         'gamma': 1,
         'seed': 12,
         # 'reg_alpha':0,
         'subsample':0.65,
         'objective':'binary:logistic'}
# print(preds)

X, y = train_datas, train_label
from sklearn import metrics
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
auc_list = []
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    bst = xgb.train(param, data_dmatrix, 10)
    test_dmatrix = xgb.DMatrix(data=X_test)
    preds = bst.predict(test_dmatrix)
    preds = np.array(preds)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    s1 = metrics.roc_auc_score(y_test, preds)
    f2 = metrics.confusion_matrix(y_test, preds)
    f3 = metrics.f1_score(y_test, preds, average='macro')
    print('Auc', s1)  # \nConfusionMatrix\n', f2, '\nF1Score', f3) 
    auc_list.append(s1)
print('Mean:%.3f' % np.mean(auc_list))

data_dmatrix = xgb.DMatrix(data=train_datas, label=train_label)
bst = xgb.train(param, data_dmatrix, 10)
# make prediction
test_dmatrix = xgb.DMatrix(data=test_datas11)
preds = bst.predict(test_dmatrix)

ID = test_datas1.iloc[:, 0].values
saved = np.hstack((ID.reshape(-1, 1), preds.reshape(-1, 1)))
np.savetxt('a.csv', saved, fmt='%d,%.2f')
