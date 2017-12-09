
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
learning_rate=0.001
training_iters=5000
batch_size=128
display_step=100

# Network Parameters
n_input=6
n_steps=6
n_hidden=128
n_classes=2

# tf Graph input
x=tf.placeholder("float",[None,n_steps,n_input])
# x=tf.placeholder("float",[None,n_steps])
y=tf.placeholder("float",[None,n_classes])

# Define Weights
weights={
    "out":tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
    # "in":tf.Variable(tf.random_uniform([n_steps,n_hidden],0,0)),
    # "out":tf.Variable(tf.random_uniform([n_hidden,n_classes],0,0))
}
biases={
    # "in":tf.Variable(tf.random_normal([n_hidden])),
    "out":tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x,weights,biases):
    x=tf.unstack(x,n_steps,1)
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    # outputs,states=rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    return tf.matmul(outputs[-1],weights["out"])+biases["out"]


pred=tf.nn.softmax(RNN(x,weights,biases))
# weight_in=tf.matmul(x,weights["in"])+biases["in"]
# pred=tf.nn.softmax(tf.matmul(weight_in,weights["out"])+biases["out"])

# Optimizer
# cost=-Gini.normalized_score(y.eval(),pred.eval())
cost=tf.reduce_sum(-y*tf.log(pred))
# cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
# cost=-tf.reduce_sum(tf.reduce_sum(y[:,1])*y[:,0]*tf.log(pred[:,0]+tf.reduce_sum(y[:,0])*y[:,1]*tf.log(pred[:,1])))
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
    # current_batch=0

    while step<training_iters:
        # if current_batch>len(x_train):
        #     current_batch-=len(x_train)
        batch_x,batch_y=next_batch(x_train_0,x_train_1,y_train_0,y_train_1,batch_size,0.1)
        batch_x=batch_x.reshape((batch_size,n_steps,n_input))
        if step % display_step==0:
            cost_=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            y_pred=sess.run(pred,feed_dict={x:batch_x})[:,1]
            norm_gini=Gini.normalized_score(np.argmax(batch_y,axis=1),y_pred)
            print "Iter "+str(step)+",Minibatch gini="+"{:6f}".format(norm_gini)+",Training Accuracy="+"{:.5f}".format(acc)+\
                  ",Training Cost="+"{:.5f}".format(cost_)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        step+=1
        # current_batch+=batch_size
    print "Optimization Finished"





    # Train
    train_x=[]
    for i in range(len(x_train)):
        item=list(x_train[i])
        # item.extend([0,0,0])
        train_x.append(item)

    train_x=np.array(train_x).reshape(len(train_x),n_steps,n_input)
    y_pred=sess.run(pred,feed_dict={x:train_x})[:,1]
    norm_gini=Gini.normalized_score(y_train,y_pred)
    print "Train Norm Gini = %f" % norm_gini

    # Test

    test_x=[]
    for i in range(len(x_test)):
        item=list(x_test[i])
        # item.extend([0,0,0])
        test_x.append(item)

    test_x=np.array(test_x).reshape((len(test_x),n_steps,n_input))
    y_pred=sess.run(pred,feed_dict={x:test_x})[:,1]

    # Create a submission file
    sub = pd.DataFrame()
    sub['id'] = id_test
    sub['target'] = y_pred
    sub.to_csv('submit/rnn_submit_'+str(norm_gini)+'.csv', index=False)



