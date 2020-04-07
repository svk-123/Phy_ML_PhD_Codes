from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras

import numpy as np
import pickle
import sys
import math
from matplotlib import pyplot as plt
#tf.enable_eager_execution()

'''
solve d2u/dx2=0 by using custom loss function
and BC
'''


# Load Data
#load data
xtmp=[]
ytmp=[]
reytmp=[]
utmp=[]
vtmp=[]
ptmp=[]
    
for ii in range(1):
    #x,y,Re,u,v
    with open('./cavity_Re100.pkl', 'rb') as infile:
        result = pickle.load(infile,encoding='bytes')
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    reytmp.extend(result[2])
    utmp.extend(result[3])
    vtmp.extend(result[4])
    ptmp.extend(result[5])   
        
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
utmp=np.asarray(utmp)
vtmp=np.asarray(vtmp)
ptmp=np.asarray(ptmp) 
           
x = xtmp[:,None] # NT x 1
y = ytmp[:,None] # NT x 1
    
u = utmp[:,None] # NT x 1
v = vtmp[:,None] # NT x 1
p = ptmp[:,None] # NT x 1
    
######################################################################
######################## Noiseles Data ###############################
######################################################################
# Training Data    
np.random.seed(123)
N_train=500
idx = np.random.choice(len(xtmp), N_train, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]
p_train = p[idx,:]

xtr=np.concatenate((x_train,y_train),axis=1)    
ttr=np.concatenate((u_train,v_train,p_train),axis=1)   

xtr = xtr.astype('float32')
ttr = ttr.astype('float32')

#xtr=x_train
#ttr=u_train


batch_size=500
dataset = tf.data.Dataset.from_tensor_slices((xtr,ttr))
dataset = dataset.shuffle(batch_size * 1).batch(batch_size)

#................#
num_epochs = 10000
#learning_rate = 0.001
#...................#
avg_loss = 0
#.................#
class myModel(tf.keras.Model):

    def __init__(self):
        super(myModel, self).__init__()

        self.fc1 = tf.keras.layers.Dense(100 ,activation='tanh')
        self.fc2 = tf.keras.layers.Dense(100 ,activation='tanh')
        self.fc3 = tf.keras.layers.Dense(100 ,activation='tanh')
        self.fc4 = tf.keras.layers.Dense(100 ,activation='tanh')
        self.fc5 = tf.keras.layers.Dense(100 ,activation='tanh')
        self.fc6 = tf.keras.layers.Dense(100 ,activation='tanh')
        self.fc7 = tf.keras.layers.Dense(100 ,activation='tanh')
        self.fc8 = tf.keras.layers.Dense(100 ,activation='tanh')
        self.fc9 = tf.keras.layers.Dense(3  ,activation='linear')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        return x
 
model = myModel()
model.build(input_shape=(None,2))
model.summary()
optimizer = tf.optimizers.Adam(learning_rate=0.001,decay=1e-5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=100, min_lr=0.0001)

def loss_object_mse(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def loss_object_gov(u,v,p,u_x,u_y,v_x,v_y,p_x,p_y,u_xx,u_yy,v_xx,v_yy):                                     
    nu=tf.constant(0.01)
    f_c =  u_x + v_y
    f_u =  u*u_x + v*u_y  + p_x - nu*(u_xx + u_yy) 
    f_v =  u*v_x + v*v_y  + p_y - nu*(v_xx + v_yy)
    
    return tf.reduce_mean(tf.square(f_c)) + tf.reduce_mean(tf.square(f_u)) + tf.reduce_mean(tf.square(f_v))
    
    
for epoch in range(num_epochs):

    for step, (x,y) in enumerate(dataset):
        
        #print (x.shape,y.shape)
        with tf.GradientTape(persistent=True) as tape:
        #with tf.GradientTape() as tape:  
            
            tape.watch(x)

            pred = model(x)
            
            u=pred[:,0:1]
            v=pred[:,1:2]     
            p=pred[:,2:3]
            
                   
            u_x=tape.gradient(u, x)[:,0] 
            u_y=tape.gradient(u, x)[:,1] 
            
            v_x=tape.gradient(v, x)[:,0] 
            v_y=tape.gradient(v, x)[:,1]      

            p_x=tape.gradient(p, x)[:,0] 
            p_y=tape.gradient(p, x)[:,1] 
            
          
            u_xx=tape.gradient(u_x, x)[:,0]
            u_yy=tape.gradient(u_y, x)[:,1]
            v_xx=tape.gradient(v_x, x)[:,0]
            v_yy=tape.gradient(v_y, x)[:,1]            
            
            loss_1 = loss_object_mse(pred,y)
            loss_2 = loss_object_gov(u,v,p,u_x,u_y,v_x,v_y,p_x,p_y,\
                                     u_xx,u_yy,v_xx,v_yy)
            
            loss=loss_1+loss_2
                            
        gradients = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch) % 1 == 0:
        print('loss',epoch,loss.numpy(),loss_1.numpy(),loss_2.numpy())        


#testing
pred = model(xtr)    
#plot-1
plt.figure(figsize=(6, 5), dpi=100)
plt.tricontourf(xtr[:,0],xtr[:,1],pred[:,0])
#plt.legend()
#plt.savefig('n_l2_n30_relu20ex.png')
plt.show()


#plot-1
plt.figure(figsize=(6, 5), dpi=100)
plt.tricontourf(xtr[:,0],xtr[:,1],ttr[:,0])
#plt.legend()
#plt.savefig('n_l2_n30_relu20ex.png')
plt.show()





