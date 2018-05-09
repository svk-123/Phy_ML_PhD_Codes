#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

@author: vinoth
"""

import time
start_time = time.time()

# Python 3.5
import numpy as np
import matplotlib.pyplot as plt
import pandas
from os import listdir
from os.path import isfile, join

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD,RMSprop, Adam, Adadelta, Adagrad, Nadam
from keras.layers import merge, Input, dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import cPickle as pickle
import pandas
import math
from rbflayer import RBFLayer, InitCentersRandom, InitCentersKmeans

import os, shutil
folder = './model_nn/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

N = 1000
#np.random.seed(29)
X = np.random.random((N,1))*1
X=np.reshape(X,N)
noise=np.random.normal(0,0.1,N)
X=X+noise
Y=np.zeros(len(X))
for i in range(len(X)):
    Y[i] = 0.2+0.4*X[i]**2+0.3*X[i]*math.sin(15*X[i])+0.05*math.cos(50*X[i])

## Training sets
xtr0 = X
ttr1 = Y

# Multilayer Perceptron
# create model
aa=Input(shape=(1,))
xx=Dense(10, activation='tanh')(aa)
#xx=Dense(10, activation='tanh')(xx)
g =Dense(1, activation='linear')(xx)

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[g])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, mode='min',verbose=1 ,patience=300, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=500, verbose=1, mode='auto')

filepath="./model_nn/model_sf_{epoch:02d}_{loss:.6f}_{val_loss:.6f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=500)

# Compile model
opt = Adam(lr=2.0e-2)

model.compile(loss= 'mean_squared_error',optimizer= opt)

hist = model.fit([xtr0], [ttr1], validation_split=0.1,\
                 epochs=5000, batch_size=64,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=False)

#save model
model.save('./model_nn/final_sf.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))



val_x=np.linspace(0,1,100)
val_y=val_x.copy()

for i in range(len(val_x)):
    val_y[i] = 0.2+0.4*val_x[i]**2+0.3*val_x[i]*math.sin(15*val_x[i])+0.05*math.cos(50*val_x[i])
y_pred = model.predict(val_x)

plt.figure()   
plt.plot(val_x, y_pred,label='pred')
plt.plot(val_x, val_y,label='true')
plt.legend()
#plt.plot([-1,1], [0,0], color='black')
#plt.xlim([-1,1])
plt.savefig('nn')
plt.show()

nn_loss=hist.history.values()[0]
nn_valloss=hist.history.values()[2]

mytime=time.time() - start_time
name=['0-val_x','1-val_y','2-y_pred','3-nn_loss','4-nn_valloss','5-mytime']
data_nn=[val_x,val_y,y_pred,nn_loss,nn_valloss,mytime,name]
with open('nn_layer1.pkl', 'wb') as outfile:
    pickle.dump(data_nn, outfile, pickle.HIGHEST_PROTOCOL)

#plt.figure()
#plt.scatter(X,Y)
#plt.show()
















