import numpy as np
import random
import tensorflow as tf
import Gini


class ConvolutionalNeuralNetwork:
    def __init__(self,x_test,learning_rate,training_iters,batch_size,display_step,n_input,n_classes):
        self.learning_rate=learning_rate
        self.training_iters=training_iters
        self.batch_size=batch_size
        self.display_step=display_step
        self.n_input=n_input
        self.n_classes=n_classes
        self.x_test=x_test


    # set weight&bias initial
    def weight_variable(self,shape):
        initial=tf.truncated_normal(shape,stddev=0.1)
        # initial=tf.constant(0.0,shape=shape)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial=tf.constant(0.1,shape=shape)
        # initial=tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)
    # Convolutional Functions
    def conv2d(self,x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    def next_batch(self,X,y):
        x_batch=[]
        y_batch=[]
        for i in range(self.batch_size):
            item=np.random.choice(len(X))
            x_batch.append(X[item])
            y_batch.append([y[item]])
        return np.array(x_batch),np.array(y_batch)




    def fit(self,x_train,y_train):

        # tf Graph input
        x=tf.placeholder("float",[None,self.n_input])
        y=tf.placeholder("float",[None,self.n_classes])

        # First Layer
        x_image=tf.reshape(x,[-1,6,6,1])
        W_cov1=self.weight_variable([3,3,1,32])
        b_conv1=self.bias_variable([32])
        h_conv1=tf.nn.relu(self.conv2d(x_image,W_cov1)+b_conv1)
        h_pool1=self.max_pool_2x2(h_conv1)

    
        # Full Connection
        W_fc1=self.weight_variable([3*3*32,128])
        b_fc1=self.bias_variable([128])
        h_pool1_flat=tf.reshape(h_pool1,[-1,3*3*32])
        h_fc1=tf.nn.relu(tf.matmul(h_pool1_flat,W_fc1)+b_fc1)

        # Dropout
        keep_prob=tf.placeholder("float")
        h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

        # Softmax
        W_fc2=self.weight_variable([128,1])
        b_fc2=self.bias_variable([1])
        pred=tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


        # Optimizer
        cost=tf.reduce_sum(tf.square(y-pred))
        optimizer=tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Evaluate model
        correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


        # Initial
        init=tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # saver.restore(sess, "bi-rnn_model_.ckpt")
            step=1

            while step<self.training_iters:
                batch_x,batch_y=self.next_batch(x_train,y_train)
                if step % self.display_step==0:
                    cost_=sess.run(cost,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
                    acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
                    y_pred=sess.run(pred,feed_dict={x:batch_x,keep_prob:1.0})
                    norm_gini=Gini.normalized_score(np.argmax(batch_y,axis=1),y_pred)
                    print "Iter "+str(step)+",Minibatch gini="+"{:6f}".format(norm_gini)+",Training Accuracy="+"{:.5f}".format(acc)+\
                  ",Training Cost="+"{:.5f}".format(cost_)
                sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})
                step+=1
            print "Optimization Finished"

            train_y_pred=sess.run(pred,feed_dict={x:x_train,keep_prob:1.0})
            test_y_pred=sess.run(pred,feed_dict={x:self.x_test,keep_prob:1.0})
            del sess


        return train_y_pred,test_y_pred