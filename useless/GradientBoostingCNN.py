from NeuralNetworks import ConvolutionalNeuralNetwork
import pandas as pd
import Gini

def gradient_boosting(nIter,shrinkage,train_x,train_y):

    cnn=ConvolutionalNeuralNetwork(x_test=x_test,learning_rate=0.01,training_iters=100
                               ,batch_size=128,display_step=100,n_input=36,n_classes=1)
    train_y_pred,test_y_pred=cnn.fit(train_x,train_y)
    del cnn
    print 1
    test_pred=test_y_pred
    train_pred=train_y_pred

    for i in range(nIter):
        print 2
        residual= train_y-train_y_pred
        # cnn=ConvolutionalNeuralNetwork(x_test=x_test,learning_rate=0.01,training_iters=100
        #                        ,batch_size=128,display_step=100,n_input=36,n_classes=1)
        train_y_pred,test_y_pred=cnn.fit(train_x,residual)
        del cnn
        print 3
        train_pred+=shrinkage*train_y_pred
        test_pred+=shrinkage*test_y_pred

    return train_pred,test_pred





train_df = pd.read_csv('Data/train.csv', na_values="-1")
test_df = pd.read_csv('Data/test.csv', na_values="-1")
id_test = test_df['id'].values

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

col_to_drop = train_df.columns[train_df.columns.str.startswith('ps_calc_')]
train_df = train_df.drop(col_to_drop, axis=1)
test_df = test_df.drop(col_to_drop, axis=1)

y_train = train_df['target'].values
x_train = train_df.drop(['target', 'id','ps_car_10_cat'], axis=1).values
x_test = test_df.drop(['id','ps_car_10_cat'], axis=1).values




nIter=5
shrinkage=0.1
train_pred,test_pred=gradient_boosting(nIter,shrinkage,x_train,y_train)
# cnn=ConvolutionalNeuralNetwork(x_test=x_test,learning_rate=0.01,training_iters=100
#                                ,batch_size=128,display_step=100,n_input=36,n_classes=1)
# train_y_pred,test_y_pred=cnn.fit(x_train,y_train)
print Gini.normalized_score(y_train,train_pred)


# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = test_pred
sub.to_csv('submit/GB_CNN_submit.csv', index=False)