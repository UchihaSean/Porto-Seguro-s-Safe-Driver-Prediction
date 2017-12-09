import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn import model_selection
from lightgbm import LGBMClassifier


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

train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')



# Preprocessing
id_test = test_df['id'].values
train_y = train_df['target'].values

train_X = train_df.drop(['target','id'], axis = 1)
test_X = test_df.drop(['id'], axis = 1)


col_to_drop = train_X.columns[train_X.columns.str.startswith('ps_calc_')]
train_X = train_X.drop(col_to_drop, axis=1)
test_X = test_X.drop(col_to_drop, axis=1)


train_X = train_X.replace(-1, 999)
test_X = test_X.replace(-1, 999)


cat_features = [a for a in train_X.columns if a.endswith('cat')]

temp=[]
for column in cat_features:
	temp = pd.get_dummies(pd.Series(train_X[column]))
	train_X = pd.concat([train_X,temp],axis=1)
	train_X = train_X.drop([column],axis=1)

for column in cat_features:
	temp = pd.get_dummies(pd.Series(test_X[column]))
	test_X = pd.concat([test_X,temp],axis=1)
	test_X = test_X.drop([column],axis=1)



class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res


xgb1=XGBClassifier(n_estimators=100,
                  max_depth=6,
                  objective="binary:logistic",
                  learning_rate=0.1,
                  subsample=.8,
                  min_child_weight=6,
                  colsample_bytree=.8,
                  scale_pos_weight=1.6,
                  gamma=10,
                  reg_alpha=8,
                  reg_lambda=1.3,max_delta_step=1,random_state=101)
xgb2=XGBClassifier(n_estimators=150,
                  max_depth=4,
                  objective="binary:logistic",
                  learning_rate=0.1,
                  subsample=.8,
                  min_child_weight=6,
                  colsample_bytree=.8,
                  scale_pos_weight=1.6,
                  gamma=10,
                  reg_alpha=8,
                  reg_lambda=1.3,max_delta_step=1,random_state=101)

xgb3=XGBClassifier(n_estimators=200,
                  max_depth=2,
                  objective="binary:logistic",
                  learning_rate=0.1,
                  subsample=.8,
                  min_child_weight=6,
                  colsample_bytree=.8,
                  scale_pos_weight=1.6,
                  gamma=10,
                  reg_alpha=8,
                  reg_lambda=1.3,max_delta_step=1,random_state=101)
rf1 = RandomForestClassifier(n_estimators=250,
                                    max_depth=12,
                                    min_samples_leaf=9,
                                    random_state=0)

rf2 = RandomForestClassifier(n_estimators=150,
                                    max_depth=8,
                                    min_samples_leaf=8,
                                    random_state=0)



log_model = LogisticRegression()



stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (xgb1,rf1,rf2))

# CV

# K ford
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
#
# cv_gini=[]
# for i in range(k_ford_number):
#         cv_vali_y_pred=stack.fit_predict(np.array(cv_train_X[i]),np.array(cv_train_y[i]),cv_vali_X[i])
#         cv_gini.append(normalized_score(cv_vali_y[i],cv_vali_y_pred))
#
# print np.mean(cv_gini)
# exit()




y_pred = stack.fit_predict(train_X, train_y, test_X)






sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('submit/stacked_4.csv', index=False)

# 0 xgb123
# 2 xgb12 rf1 2==200
# 3 xgb12 rf1   2==150
# 4 xgb1 rf12