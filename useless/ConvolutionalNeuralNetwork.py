
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
import Gini
import random

def next_batch(x_0,x_1,y_0,y_1,batch_size,radio):
    x_batch=[]
    y_batch=[]
    item_0=range(len(y_0))
    item_1=range(len(y_1))
    for i in range(batch_size):
        rd=random.random()
        if rd>=radio:
            item=random.choice(item_0)
            x_batch.append(x_0[item])
            y_batch.append([1,0])
        else:
            item=random.choice(item_1)
            x_batch.append(x_1[item])
            y_batch.append([0,1])
    return np.array(x_batch),np.array(y_batch)



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
y_train_0=[]
y_train_1=[]
x_train_0=[]
x_train_1=[]
for i in range(len(y_train)):
    if y_train[i]==1:
        item=list(x_train[i])
        # item.extend([0,0,0])
        x_train_1.append(item)
        y_train_1.append(1)
    else:
        item=list(x_train[i])
        # item.extend([0,0,0])
        x_train_0.append(item)
        y_train_0.append(0)


print "Data prepared"


# Parameters
learning_rate=0.01
training_iters=1000
batch_size=1280
display_step=100

# Network Parameters
n_input=36
n_classes=2

# tf Graph input
x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_classes])

# set weight&bias initial
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    # initial=tf.constant(0.0,shape=shape)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    # initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
# Convolutional Functions
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


# First Layer
x_image=tf.reshape(x,[-1,6,6,1])
W_cov1=weight_variable([3,3,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_cov1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)


# Full Connection
W_fc1=weight_variable([3*3*32,128])
b_fc1=bias_variable([128])
h_pool1_flat=tf.reshape(h_pool1,[-1,3*3*32])
h_fc1=tf.nn.relu(tf.matmul(h_pool1_flat,W_fc1)+b_fc1)

# Dropout
keep_prob=tf.placeholder("float")
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

# Softmax
W_fc2=weight_variable([128,2])
b_fc2=bias_variable([2])
pred=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


# Optimizer
cost=tf.reduce_sum(-y*tf.log(pred))
optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# Answer
# answer=tf.arg_max(pred,1)

# Save
saver = tf.train.Saver()

# Initial
init=tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, "bi-rnn_model_.ckpt")
    step=1

    while step<training_iters:
        batch_x,batch_y=next_batch(x_train_0,x_train_1,y_train_0,y_train_1,batch_size,0.2)
        if step % display_step==0:
            cost_=sess.run(cost,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
            y_pred=sess.run(pred,feed_dict={x:batch_x,keep_prob:1.0})[:,1]
            norm_gini=Gini.normalized_score(np.argmax(batch_y,axis=1),y_pred)
            print "Iter "+str(step)+",Minibatch gini="+"{:6f}".format(norm_gini)+",Training Accuracy="+"{:.5f}".format(acc)+\
                  ",Training Cost="+"{:.5f}".format(cost_)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})
        step+=1
    print "Optimization Finished"





    # Train
    train_x=[]
    for i in range(len(x_train)):
        item=list(x_train[i])
        train_x.append(item)

    # train_x=np.array(train_x).reshape(len(train_x),n_steps,n_input)
    y_pred=sess.run(pred,feed_dict={x:train_x,keep_prob:1.0})[:,1]
    norm_gini=Gini.normalized_score(y_train,y_pred)
    print "Train Norm Gini = %f" % norm_gini

    # Test
    test_x=[]
    for i in range(len(x_test)):
        item=list(x_test[i])
        # item.extend([0,0,0])
        test_x.append(item)

    # test_x=np.array(test_x).reshape((len(test_x),n_steps,n_input))
    y_pred=sess.run(pred,feed_dict={x:test_x,keep_prob:1.0})[:,1]

    # Create a submission file
    sub = pd.DataFrame()
    sub['id'] = id_test
    sub['target'] = y_pred
    sub.to_csv('submit/cnn_submit_'+str(norm_gini)+'.csv', index=False)



