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


from main_tau import L,T,bD


# ---------ML PART:-----------#
#shuffle data
N= len(L)
I = np.arange(N)
np.random.shuffle(I)
n=1200
'''
## Training sets
xtr0 = L[I][:n]
xtr1 = T[I][:n][0]
xtr2 = T[I][:n][1]
xtr3 = T[I][:n][2]
xtr4 = T[I][:n][3]
xtr5 = T[I][:n][4]
xtr6 = T[I][:n][5]

ttr1 = bD[I][:n][0]
ttr2 = bD[I][:n][1]
ttr3 = bD[I][:n][2]
ttr4 = bD[I][:n][3]
ttr5 = bD[I][:n][4]
ttr6 = bD[I][:n][5]
'''

xtr0 = L

xtr1 = T[:,:,0]
xtr2 = T[:,:,1]
xtr3 = T[:,:,2]
xtr4 = T[:,:,4]
xtr5 = T[:,:,5]
xtr6 = T[:,:,8]

ttr1 = bD[:,0]
ttr2 = bD[:,1]
ttr3 = bD[:,2]
ttr4 = bD[:,4]
ttr5 = bD[:,5]
ttr6 = bD[:,8]



# Multilayer Perceptron
# create model
aa=Input(shape=(5,))
xx =Dense(10,  kernel_initializer='random_normal', activation='relu')(aa)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
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
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, mode='min',verbose=1 ,patience=100, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='val_loss', min_delta=1.0e-6, patience=1000, verbose=1, mode='auto')

filepath="./model/model_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=1000)

#os.system("rm ./graph/* ")
#tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

# Compile model
opt = Adam(lr=2.5e-4,decay=1.0e-9)

model.compile(loss= 'mean_squared_error',optimizer= opt)


hist = model.fit([xtr0,xtr1,xtr2,xtr3,xtr4,xtr5,xtr6], [ttr1,ttr2,ttr3,ttr4,ttr5,ttr6], validation_split=0.2,\
                 epochs=10000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=True)

#hist = model.fit([xtr0,xtr1], [ttr1], validation_split=0.2,\
#                 epochs=30000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1,shuffle=True)


#save model
model.save('./model/final.hdf5') 


print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))























