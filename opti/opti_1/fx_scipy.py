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
import cPickle as pickle
from sklearn import preprocessing
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
import math 
from scipy.optimize import minimize

N = 5000
np.random.seed(29)
X = np.random.random((N,1))*1
X=np.reshape(X,N)
noise=np.random.normal(0,0.1,N)
X=X+noise
Y=np.zeros(len(X))
for i in range(len(X)):
    Y[i] = 0.2+0.4*X[i]**2+0.3*X[i]*math.sin(15*X[i])+0.05*math.cos(50*X[i])

#plt.show()

#  Splitting Data
I = np.arange(N)
np.random.shuffle(I)
n =N

## Training sets
xtr = X[I][:n]
ttr = Y[I][:n]
## Testing sets
xte = X[I][n:]
tte = Y[I][n:]

aa=Input(shape=(1,))
xx =Dense(10, kernel_initializer='random_normal')(aa)
xx=LeakyReLU(alpha=.1)(xx)
g =Dense(1, activation='linear')(xx)

#model
model = Model(inputs=[aa], outputs=[g])

def loss(W):
    weightsList = [np.zeros((1,10)), np.zeros(10), np.zeros((10,1)), np.zeros(1)]
    weightsList[0][0,0:10] = W[0:10]
    weightsList[1][0:10] = W[10:20]
    weightsList[2][0:10,0] = W[20:30]
    weightsList[3][0] = W[30]    
    
    model.set_weights(weightsList)
    preds = model.predict(X)
    mse = np.sum(np.square(np.subtract(preds,Y)))/len(X)
    return mse

V = np.random.normal(-1,1,31)
print('Starting loss = {}'.format(loss(V)))
# set the eps option to increase the epsilon used in numerical diff
bnds=((-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),\
      (-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),\
      (-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),(-3,3),\
      (-3,3))

res = minimize(loss, x0=V, method = 'L-BFGS-B', bounds=bnds, \
               options={'disp': True, 'maxcor': 100, 'ftol': 1 * np.finfo(float).eps, \
                                 'eps': 1e-07, 'maxfun': 50000, \
                                 'maxiter': 50000, 'maxls': 100})
print('Ending loss = {}'.format(loss(res.x)))

xt=np.linspace(0,1.2,100)
yt=np.zeros(len(xt))
for i in range(len(xt)):
    yt[i] = 0.2+0.4*xt[i]**2+0.3*xt[i]*math.sin(15*xt[i])+0.05*math.cos(50*xt[i])

z=np.linspace(0,1.0,100)
zt=np.zeros(len(z))
for i in range(len(zt)):
    zt[i] = 0.2+0.4*z[i]**2+0.3*z[i]*math.sin(15*z[i])+0.05*math.cos(50*z[i])

ze=np.linspace(1.0,1.2,100)
zte=np.zeros(len(ze))
for i in range(len(zte)):
    zte[i] = 0.2+0.4*ze[i]**2+0.3*ze[i]*math.sin(15*ze[i])+0.05*math.cos(50*ze[i])
    
pred = model.predict(xt)    
predz = model.predict(z)
predze = model.predict(ze)

#plot-1
plt.figure(figsize=(6, 5), dpi=100)
#l1,=plt.plot(X,Y,'o',label='training pts')
#l1,=plt.plot(xt,yt,label='f(x)')
#l1,=plt.plot(xt,pred,label='pred')
l1,=plt.plot(z,predz,label='NN')
l1,=plt.plot(xt,yt,label='true')
l1,=plt.plot(ze,predze,label='NN-ext')

plt.legend()
plt.savefig('n_l2_n30_relu20ex.png')
plt.show()




























