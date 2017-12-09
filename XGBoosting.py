from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import decomposition
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def score(actual,predicted):
    assert len(actual)==len(predicted)

    all=[]
    for i in range(len(actual)):
        all.append([actual[i],predicted[i]])

    sorted_all=sorted(all,key=lambda x:x[1],reverse=True)
    actual_sum=np.sum(sorted_all,axis=0)[0]
    current_item=0
    sum_item=0
    for item in sorted_all:
        current_item+=item[0]/(actual_sum+0.0)
        sum_item+=current_item
    sum_item-=(len(actual)+1)/2.0

    return sum_item/len(actual)

def normalized_score(actual,predicted):
    return score(actual,predicted)/score(actual,actual)


train_df = pd.read_csv('input/train.csv', na_values="-1")
test_df = pd.read_csv('input/test.csv', na_values="-1")
id_test = test_df['id'].values

col_to_drop = train_df.columns[train_df.columns.str.startswith('ps_calc_')]
train_df = train_df.drop(col_to_drop, axis=1)
test_df = test_df.drop(col_to_drop, axis=1)

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

cat_features = [a for a in train_df.columns if a.endswith('cat')]

temp=[]
for column in cat_features:
	temp = pd.get_dummies(pd.Series(train_df[column]))
	train_df = pd.concat([train_df,temp],axis=1)
	train_df = train_df.drop([column],axis=1)

for column in cat_features:
	temp = pd.get_dummies(pd.Series(test_df[column]))
	test_df = pd.concat([test_df,temp],axis=1)
	test_df = test_df.drop([column],axis=1)

train_X=train_df.drop(['target','id'],axis=1).values
train_y=train_df['target'].values
test_X=test_df.drop(['id'],axis=1).values

print (len(train_X[0]))


# # Scale
# scale=preprocessing.StandardScaler().fit(train_X)
# train_X=scale.transform(train_X)
# test_X=scale.transform(test_X)
#
# # PCA
# pca=decomposition.PCA(n_components=199,random_state=101)
# train_X=pca.fit_transform(train_X)
# print (pca.explained_variance_ratio_)
# test_X=pca.transform(test_X)
# print (len(train_X[0]))



# # CV
#
# # K ford
# k_ford_number=3
# kf=model_selection.KFold(n_splits=k_ford_number,random_state=101)
# cv_train_X=[]
# cv_train_y=[]
# cv_vali_X=[]
# cv_vali_y=[]
# for i in range(k_ford_number):
#     cv_train_X.append([])
#     cv_train_y.append([])
#     cv_vali_X.append([])
#     cv_vali_y.append([])
#
# for num,(train_index,vali_index) in enumerate(kf.split(train_X)):
#     for index in train_index:
#         cv_train_X[num].append(train_X[index])
#         cv_train_y[num].append(train_y[index])
#     for index in vali_index:
#         cv_vali_X[num].append(train_X[index])
#         cv_vali_y[num].append(train_y[index])


# hyperparameter
estimator_set=np.arange(500,800,50)
estimated_gini=[]
max_depth_set=[2,4,6]
learning_rate_set=[0.1,0.01,0.001]

best_estimator=400
best_max_depth=4
best_learning_rate=0.1

# for estimator in estimator_set:
# # for max_depth in max_depth_set:
# # for learning_rate in learning_rate_set:
#     cv_gini=[]
#     xgb=XGBClassifier(n_estimators=estimator,
#                   max_depth=best_max_depth,
#                   objective="binary:logistic",
#                   learning_rate=best_learning_rate,
#                   subsample=.8,
#                   min_child_weight=6,
#                   colsample_bytree=.8,
#                   scale_pos_weight=1.6,
#                   gamma=10,
#                   reg_alpha=8,
#                   reg_lambda=1.3,max_delta_step=1,random_state=101)
#     for i in range(k_ford_number):
#         xgb.fit(np.array(cv_train_X[i]),np.array(cv_train_y[i]))
#         cv_vali_y_pred=xgb.predict_proba(cv_vali_X[i])[:,1]
#         cv_gini.append(normalized_score(cv_vali_y[i],cv_vali_y_pred))
#     print (estimator,np.mean(cv_gini))
#     estimated_gini.append(np.mean(cv_gini))
#
# print(estimated_gini)
#
# plt.plot(estimator_set,estimated_gini)
# plt.xlabel("n_estimator")
# plt.ylabel("cv normalized gini score")
# plt.title("xgb n_estimator choices with drop features")
# plt.show()
#
#
# exit()

# LB
xgb=XGBClassifier(n_estimators=best_estimator,
                  max_depth=best_max_depth,
                  objective="binary:logistic",
                  learning_rate=best_learning_rate,
                  subsample=.8,
                  min_child_weight=6,
                  colsample_bytree=.8,
                  scale_pos_weight=1.6,
                  gamma=10,
                  reg_alpha=8,
                  reg_lambda=1.3,max_delta_step=1,random_state=101)

xgb.fit(train_X,train_y)
train_y_pred=xgb.predict_proba(train_X)[:,1]
train_gini= normalized_score(train_y,train_y_pred)
print (train_gini)

test_y_pred=xgb.predict_proba(test_X)[:,1]

# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = test_y_pred
sub.to_csv('submit/xgb_0.282.csv', index=False)


