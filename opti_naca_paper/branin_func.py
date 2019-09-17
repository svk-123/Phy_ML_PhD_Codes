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

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers import merge, Input, dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import pickle
from sklearn import preprocessing
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
import math

aa=np.linspace(-5,10,100)
bb=np.linspace(0,15,100)

X1,X2 = np.meshgrid(aa,bb,indexing='ij')
Y=X1.copy()
Y[:,:]=0



a=1
b=5.1/(4*np.pi**2)
c=5/np.pi
r=6
s=10
t=1/(8*np.pi)
for i in range(100):
    for j in range(100):
        x1=X1[i,j]
        x2=X2[i,j]
        
        y =  a * (x2 - (b*x1**2) + (c*x1) - r)**2 + \
        s*(1-t)*np.cos(x1) + s

        Y[i,j] = y
                       
#plt.figure()
#plt.contour(X1,X2,Y,20)
#plt.colorbar()
#plt.show()        
X1=X1/10
X2=X2/15
Y=Y/308.12909601160663


XX = np.concatenate((X1.flatten()[:,None],X2.flatten()[:,None]),axis=1)
YY = Y.flatten()[:,None]

#  Splitting Data
N=10000
I = np.arange(N)
np.random.shuffle(I)
n =100

## Training sets
xtr = XX[I][:n]
ttr = YY[I][:n]

aa=Input(shape=(2,))
xx =Dense(10, kernel_initializer='random_normal',activation='tanh')(aa)
xx =Dense(10, kernel_initializer='random_normal',activation='tanh')(xx)
g =Dense(1, activation='linear')(xx)

#model
model = Model(inputs=[aa], outputs=[g])
# Compile model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min'\
                              ,verbose=1 ,patience=200, min_lr=1.0e-6)

e_stop = EarlyStopping(monitor='val_loss', min_delta=1.0e-5, patience=500, \
                       verbose=1, mode='auto')

opt = Adam(lr=0.00001)
model.compile(loss= 'mean_squared_error', optimizer= opt)

hist = model.fit(xtr, ttr, validation_split=0.2,\
                 callbacks=[reduce_lr,e_stop],epochs=10000)
 
model.save('my_model2.h5') 
print("--- %s seconds ---" % (time.time() - start_time))

y_pred = model.predict(XX)



plt.figure(1)
plt.contour(X1,X2,Y,20)
plt.show()


plt.figure(2)
plt.tricontour(XX[:,0],XX[:,1],y_pred[:,0],20)
plt.show()


