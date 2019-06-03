from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import sys
import math
from matplotlib import pyplot as plt

#data
N = 500
np.random.seed(29)
X = np.random.random((N,1))*1
X=np.reshape(X,N)
#noise=np.random.normal(0,0.1,N)
#X=X+noise
Y=np.zeros(len(X))

for i in range(len(X)):
    Y[i] = 0.2+0.4*X[i]**2
    
    
#plt.show()

#  Splitting Data
I = np.arange(N)
np.random.shuffle(I)
n =450

xx=X.copy()
## Training sets
xtr = X[I][:n]
ttr = Y[I][:n]
xtr=np.reshape(xtr,(len(xtr),1))
ttr=np.reshape(ttr,(len(ttr),1))

## Testing sets
xte = X[I][n:]
tte = Y[I][n:]
xte=np.reshape(xte,(len(xte),1))
tte=np.reshape(tte,(len(tte),1))

def get_batch(idx,bs,tb):
    if(idx<tb):
        return xtr[idx*bs:(idx+1)*bs],ttr[idx*bs:(idx+1)*bs]
    else:
        return xtr[(idx+1)*bs:],ttr[(idx+1)*bs:]

#------------------------------------------------------------------------------
############# BUILD NN #############
## Parameters
learning_rate = 0.0001
training_epochs = 1000
batch_size = 32
display_step = 1


# tf Graph input
n_input = 1 
n_output = 1
X = tf.placeholder(tf.float64, [None, n_input],name='inputs')
Y = tf.placeholder(tf.float64, [None, n_output])

#create model
l1 = tf.layers.dense(X,  30, activation=tf.nn.relu)
l1 = tf.layers.dense(l1, 30, activation=tf.nn.relu)
l2 = tf.layers.dense(l1, 2, activation=tf.nn.relu)


output=tf.layers.dense(l2,1,activation=None,name='prediction')

# Define loss and optimizer
loss = tf.losses.mean_squared_error(Y, output)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train= optimizer.minimize(loss)

#validation loss
val_loss = tf.losses.mean_squared_error(Y, output)
pgrad_x = tf.gradients(output, X)[0]

# Initializing the variables
init = tf.global_variables_initializer()

#saver
saver = tf.train.Saver()

#session-run
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(xtr.shape[0]/batch_size)
        if((xtr.shape[0]%batch_size) != 0):
            total_batch +=1
            
        # Loop over all batches
        for i in range(total_batch):
            
            batch_x, batch_y = get_batch(i,batch_size,total_batch)
            
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train,loss], feed_dict={X: batch_x,Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        
        #validate
        v = sess.run(val_loss, feed_dict={X: xte,Y: tte})
        pgradx=sess.run(pgrad_x, feed_dict={X: xtr,Y: ttr})
        
        #v1,v2=sess.run([x1,y1])
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "trE={:.9f}".format(avg_cost), "valE={:.9f}".format(v))
    saver.save(sess,'./tf_model/model')
    
    print("Optimization Finished!")




    pred = sess.run(output, feed_dict={X: xtr,Y: ttr})     
    fig=plt.figure(1)
    plt.plot(xtr[:,0],ttr[:,0],'og',label='true')
    plt.plot(xtr[:,0],pred[:,0],'or',ms=2,label='nn')
    plt.show()

#    fig=plt.figure(2)
#    plt.plot(xtr[:,0],pgradx,'or',ms=2,label='pd')
#    plt.plot(xx,dY,'og',ms=2,label='true')
#    plt.show()
















