from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import sys


# load and process data
#load data
xtmp=[]
ytmp=[]
reytmp=[]
utmp=[]
vtmp=[]
ptmp=[]
flist=['Re1000','Re2000','Re3000','Re4000','Re5000','Re7000','Re8000','Re9000']
for ii in range(len(flist)):
    #x,y,Re,u,v
    with open('./data/cavity_%s.pkl'%flist[ii], 'rb') as infile:
        result = pickle.load(infile,encoding='bytes')
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    reytmp.extend(result[2])
    utmp.extend(result[3])
    vtmp.extend(result[4])
    ptmp.extend(result[5])   
    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
reytmp=np.asarray(reytmp)
utmp=np.asarray(utmp)
vtmp=np.asarray(vtmp)
ptmp=np.asarray(ptmp) 

# ---------ML PART:-----------#
#shuffle data
N= len(utmp)
I = np.arange(N)
np.random.shuffle(I)
n=10000

#normalize
reytmp=reytmp/10000.

my_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
my_out=np.concatenate((utmp[:,None],vtmp[:,None],ptmp[:,None]),axis=1)


## Training sets
xtr= my_inp[I][:n]
ttr = my_out[I][:n]

def get_batch(idx,bs,tb):
    if(idx<tb):
        return xtr[idx*bs:(idx+1)*bs],ttr[idx*bs:(idx+1)*bs]
    else:
        return xtr[(idx+1)*bs:],ttr[(idx+1)*bs:]

#------------------------------------------------------------------------------
############# BUILD NN #############
## Parameters
learning_rate = 0.0001
training_epochs = 100
batch_size = 1000
display_step = 1


# tf Graph input
n_input = 3 
n_output = 3 
X = tf.placeholder(tf.float64, [None, n_input])
Y = tf.placeholder(tf.float64, [None, n_output])


#create model
l1 = tf.layers.dense(X,  30, activation=tf.nn.relu)
l1 = tf.layers.dense(l1, 30, activation=tf.nn.relu)
l1 = tf.layers.dense(l1, 30, activation=tf.nn.relu)
l1 = tf.layers.dense(l1, 30, activation=tf.nn.relu)
l1 = tf.layers.dense(l1, 30, activation=tf.nn.relu)
l1 = tf.layers.dense(l1, 30, activation=tf.nn.relu)
l1 = tf.layers.dense(l1, 30, activation=tf.nn.relu)
output=tf.layers.dense(l1,3)

# Define loss and optimizer
loss = tf.losses.mean_squared_error(Y, output)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train= optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

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
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "tr_error={:.9f}".format(avg_cost))
    print("Optimization Finished!")






















