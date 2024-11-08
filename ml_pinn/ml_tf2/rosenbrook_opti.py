from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras

import numpy as np
import pickle
import sys
import math
# from matplotlib import pyplot as plt
# #tf.enable_eager_execution()
# from mpl_toolkits import mplot3d




def rosen(x):

    """The Rosenbrock function"""

    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

#print(rosen(np.asarray([0.5,0.5,0.5,0.5])))


x1=np.linspace(-0.5,1.5,500)
x2=np.linspace(-0.5,1.5,500)

a1, b1 = np.meshgrid(x1, x2)
a=a1.flatten()
b=b1.flatten()
x3=np.concatenate((a[:,None],b[:,None]),axis=1)

y1=[]
for i in range(len(x3)):
    y1.append(rosen(x3[i,:]))
y1=np.asarray(y1)



######################################################################
######################## Noiseles Data ###############################
######################################################################
# Training Data    
np.random.seed(123)
N_train=len(a)
idx = np.random.choice(len(a), N_train, replace=False)
a = a[idx]
b = b[idx]
y1 = y1[idx]


xtr=np.concatenate((a[:,None],b[:,None]),axis=1)    
ttr=y1[:,None]  

xtr = xtr.astype('float32')
ttr = ttr.astype('float32')


#xtr=x_train
#ttr=u_train

#................#
num_epochs = 10000
#learning_rate = 0.001
#...................#
avg_loss = 0
#.................#

#model
inputs = tf.keras.Input(shape=(2,))
xx = tf.keras.layers.Dense(100, activation=tf.nn.tanh)(inputs)
xx = tf.keras.layers.Dense(100, activation=tf.nn.tanh)(xx)
xx = tf.keras.layers.Dense(100, activation=tf.nn.tanh)(xx)
xx = tf.keras.layers.Dense(100, activation=tf.nn.tanh)(xx)
xx = tf.keras.layers.Dense(100, activation=tf.nn.tanh)(xx)
xx = tf.keras.layers.Dense(100, activation=tf.nn.tanh)(xx)
outputs = tf.keras.layers.Dense(1)(xx)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer = tf.optimizers.Adam(learning_rate=0.001,decay=1e-5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=10, min_lr=0.000001)

model.compile(optimizer=optimizer,loss='mse')
model.fit(xtr, ttr, batch_size=2000, epochs=1000,callbacks=reduce_lr,verbose=1)
#
model.save('rosen_hr_mlp')

'''
new_model = tf.keras.models.load_model('rosen_mlp')
print(new_model.predict(np.asarray([0.9266516305989948, 0.8578092107742337]).reshape(1,2)))

'''
