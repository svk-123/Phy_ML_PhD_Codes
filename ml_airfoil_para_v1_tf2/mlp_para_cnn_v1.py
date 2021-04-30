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

# ref:[data,name]

inp=[]
out=[]
xx=[]
name=[]

data_file='./data_file/foil_uiuc.pkl'	
                
with open(data_file, 'rb') as infile:
    result = pickle.load(infile,encoding='bytes')
    print (result[-1:])    
            
    inp.extend(result[0])
    out.extend(result[1])
    xx.extend(result[2])


inp=np.asarray(inp)
out=np.asarray(out)
out=out/0.25

I=range(len(inp))
I=np.asarray(I)
np.random.shuffle(I)

xtr1=inp[I]
ttr1=out[I]

del result
del inp
del out
del name

xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))  

# print dataset values
print('xtr shape:', xtr1.shape)
print('ttr shape:', ttr1.shape)

# Multilayer Perceptron
# create model
# construct model
#aa = Input([216,216,1])

inputs = tf.keras.Input(shape=(216,216,1))
x=tf.keras.layers.Conv2D(32, 4, activation='swish',padding='same')(inputs)
x=tf.keras.layers.MaxPool2D(pool_size=(3,3))(x)
x=tf.keras.layers.Conv2D(64, 4, activation='swish',padding='same')(x)
x=tf.keras.layers.MaxPool2D(pool_size=(3,3))(x)
x=tf.keras.layers.Conv2D(64, 4, activation='swish',padding='same')(x)
x=tf.keras.layers.MaxPool2D(pool_size=(3,3))(x)
x=tf.keras.layers.Conv2D(128, 2, activation='swish',padding='same')(x)
x=tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
x=tf.keras.layers.Conv2D(256, 2, activation='swish',padding='same')(x)
x=tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)

x=tf.keras.layers.Flatten()(x)

x=tf.keras.layers.Dense(units=200,activation=tf.nn.tanh)(x)
x=tf.keras.layers.Dense(units=200,activation=tf.nn.tanh)(x)
x=tf.keras.layers.Dense(units=200,activation=tf.nn.tanh)(x)
x=tf.keras.layers.Dense(units=8,activation=tf.nn.tanh)(x)
x=tf.keras.layers.Dense(units=200,activation=tf.nn.tanh)(x)
x=tf.keras.layers.Dense(units=200,activation=tf.nn.tanh)(x)
x=tf.keras.layers.Dense(units=200,activation=tf.nn.tanh)(x)

outputs=tf.keras.layers.Dense(units=100,activation=tf.nn.tanh)(x)






e_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=30, verbose=1, mode='auto')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=20,\
                                                 epsilon=1.0e-8, min_lr=1.0e-8)
    
filepath="./model_cnn/model_cnn_{epoch:02d}_{loss:.6f}_{val_loss:.6f}.hdf5"

chkpt= tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=10)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer = tf.optimizers.Adam(learning_rate=1e-4,decay=1e-5)

model.compile(optimizer=optimizer,loss='mse')
hist=model.fit(xtr1, ttr1, validation_split=0.1, batch_size=64, epochs=50, callbacks=[reduce_lr,chkpt],verbose=1)

model.save('./model_cnn/final')
model.save('./model_cnn/final_model.hdf5')

data1=[hist.history]
with open('./model_cnn/hist.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)
#new_model = tf.keras.models.load_model('final')
#print(new_model.predict(np.asarray(my_inp[0:10,:])))


