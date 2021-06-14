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
import matplotlib

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
from numpy import linalg as LA

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

N = 5000
#np.random.seed(29)
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
n = N

## Training sets
xtr = X[I][:n]
ttr = Y[I][:n]
## Testing sets
xte = X[I][n:]
tte = Y[I][n:]

aa=Input(shape=(1,))
xx =Dense(200, kernel_initializer='random_normal',activation='tanh')(aa)
#xx =Dense(30,activation='tanh')(xx)
#xx =Dense(30,activation='tanh')(xx)
#xx =Dense(30,activation='tanh')(xx)
#xx =Dense(50,activation='tanh')(xx)
#xx =Dense(50,activation='tanh')(xx)
g =Dense(1, activation='linear')(xx)


#model
model = Model(inputs=[aa], outputs=[g])
# Compile model
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min'\
                              ,verbose=1 ,patience=200, min_lr=1.0e-6)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-7, patience=500, \
                       verbose=1, mode='auto')

opt = Adam(lr=0.001)
model.compile(loss= 'mean_squared_error', optimizer= opt)

hist = model.fit(xtr, ttr, validation_split=0.2,\
                 callbacks=[reduce_lr,e_stop],epochs=5000)
 


suff='l4_30_tanh'
model.save('my_model_%s.h5'%suff) 
print("--- %s seconds ---" % (time.time() - start_time))

loss1=hist.history['loss']
loss2=hist.history['val_loss']
lrate=hist.history['lr']

data1=[hist.history]
with open('./hist/hist_%s.pkl'%suff, 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)


# only testing 

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
'''
mei=10
plt.figure(figsize=(6, 5), dpi=100)
plt.plot(xt,yt,'b',marker='o',mfc='None',mew=1.5,mec='blue',linewidth=3,ms=10,markevery=mei,label='True')
#plt.plot(z,zt,'g',linewidth=3,label='MLP-tr')
#plt.plot(ze,zte,'r',linewidth=3,label='MLP-ts')
plt.plot(z,predz,'g',linewidth=3,label='MLP-tr')
plt.plot(ze,predze,'r',linewidth=3,label='MLP-ts')
#plt.plot(u3d,yd,'r',linewidth=3,label='NN')
plt.legend(loc="upper left", bbox_to_anchor=[-0.02, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
#plt.yticks([])    
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)    
plt.xlabel('X',fontsize=20)
plt.ylabel('f(x)',fontsize=20)
    
#plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
#plt.savefig('./plot/u_%s.tiff'%(suff), format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()

'''






'''
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

#plot-2
plt.figure(figsize=(6, 5), dpi=100)
l1,=plt.plot(range(0,len(loss1[:800])),loss1[:800],label='Training')
l1,=plt.plot(range(0,len(loss2[:800])),loss2[:800],label='Validation')
#l1,=plt.plot(z,predf,label='NNfun')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('LS Error')
plt.savefig('c_l2_n30_relu5000.png')
plt.show()
'''

























