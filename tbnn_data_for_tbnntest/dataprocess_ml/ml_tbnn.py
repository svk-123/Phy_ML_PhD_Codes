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
import os

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
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
#


from main_tau import L,T1m,T2m,T3m,T4m,T5m,T6m,aD,bD


bD1=bD[:,0]
bD2=bD[:,1]
bD3=bD[:,2]
bD4=bD[:,3]
bD5=bD[:,4]
bD6=bD[:,5]

# ---------ML PART:-----------#
#shuffle data
N= len(L)
I = np.arange(N)
np.random.shuffle(I)
n=59
'''
## Training sets
xtr0 = L[I][:n]
xtr1 = T1m[I][:n]
xtr2 = T2m[I][:n]
xtr3 = T3m[I][:n]
xtr4 = T4m[I][:n]
xtr5 = T5m[I][:n]
xtr6 = T6m[I][:n]

ttr1 = bD1[I][:n]
ttr2 = bD2[I][:n]
ttr3 = bD3[I][:n]
ttr4 = bD4[I][:n]
ttr5 = bD5[I][:n]
ttr6 = bD6[I][:n]
'''

xtr0 = L
xtr1 = T1m
xtr2 = T2m
xtr3 = T3m
xtr4 = T4m
xtr5 = T5m
xtr6 = T6m

ttr1 = bD1
ttr2 = bD2
ttr3 = bD3
ttr4 = bD4
ttr5 = bD5
ttr6 = bD6



# Multilayer Perceptron
# create model
aa=Input(shape=(5,))
xx =Dense(20,  kernel_initializer='random_normal', activation='tanh')(aa)
xx =Dense(20, activation='tanh')(xx)
g =Dense(10, activation='linear')(xx)


t1=Input(shape=(10,))
y1= dot([g,t1], 1)

t2=Input(shape=(10,))
y2= dot([g, t2], 1)

t3=Input(shape=(10,))
y3= dot([g, t3], 1)

t4=Input(shape=(10,))
y4= dot([g, t4], 1)

t5=Input(shape=(10,))
y5= dot([g, t5], 1)

t6=Input(shape=(10,))
y6= dot([g, t6], 1)

#model save dir


#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa,t1,t2,t3,t4,t5,t6], outputs=[y1,y2,y3,y4,y5,y6])

#callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, mode='min',verbose=1 ,patience=10, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='val_loss', min_delta=1.0e-6, patience=100, verbose=1, mode='auto')

filepath="./model/model_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=5000)

os.system("rm ./graph/* ")
tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

# Compile model
opt = Adam(lr=2.50e-6,decay=1.0e-9)

model.compile(loss= 'mean_squared_error',optimizer= opt)


hist = model.fit([xtr0,xtr1,xtr2,xtr3,xtr4,xtr5,xtr6], [ttr1,ttr2,ttr3,ttr4,ttr5,ttr6], validation_split=0.2,\
                 epochs=30000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1,shuffle=True)

#save model
model.save('./model/final.hdf5') 


print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))























