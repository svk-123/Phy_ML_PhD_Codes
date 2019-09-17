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
import pandas

import os, shutil


#load data
xtmp=[]
ytmp=[]
reytmp=[]
aoatmp=[]
utmp=[]
vtmp=[]
ptmp=[]

flist=['Re100','Re200','Re500','Re1000','Re1500']
AoA=range(0,11)
foil='n0012'

for ii in range(1):
    for jj in range(1):
        #x,y,Re,u,v
        with open('./data_file/foil_naca0012_Re_100to2000_aoa_0to16_train.pkl', 'rb') as infile:
            result = pickle.load(infile)
        xtmp.extend(result[0])
        ytmp.extend(result[1])
        reytmp.extend(result[3])
        aoatmp.extend(result[4])
        ptmp.extend(result[5])
        utmp.extend(result[6])
        vtmp.extend(result[7])   
    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
reytmp=np.asarray(reytmp)
aoatmp=np.asarray(aoatmp)
utmp=np.asarray(utmp)
vtmp=np.asarray(vtmp)
ptmp=np.asarray(ptmp) 

# ---------ML PART:-----------#
#shuffle data
N= len(utmp)
I = np.arange(N)
np.random.shuffle(I)
n=122*120

#normalize
reytmp=reytmp/2500.
aoatmp=aoatmp/16.

my_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None],aoatmp[:,None]),axis=1)
my_out=np.concatenate((utmp[:,None],vtmp[:,None],ptmp[:,None]),axis=1)


## Training sets
xtr0= my_inp[I][:n]
ttr1 = my_out[I][:n]

# Multilayer Perceptron
# create model
aa=Input(shape=(4,))
xx =Dense(100,  kernel_initializer='random_normal', activation='tanh')(aa)
xx =Dense(100, activation='tanh')(xx)
xx =Dense(100, activation='tanh')(xx)
xx =Dense(100, activation='tanh')(xx)
xx =Dense(100, activation='tanh')(xx)
xx =Dense(100, activation='tanh')(xx)
xx =Dense(100, activation='tanh')(xx)
xx =Dense(100, activation='tanh')(xx)
g =Dense(3, activation='linear')(xx)

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[g])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.167, mode='min',verbose=1 ,patience=300, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=500, verbose=1, mode='auto')

filepath="./model/model_sf_{epoch:02d}_{loss:.6f}_{val_loss:.6f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=500)

# Compile model
opt = Adam(lr=2.5e-4,decay=1.0e-12)

model.compile(loss= 'mean_squared_error',optimizer= opt)

hist = model.fit([xtr0], [ttr1], validation_split=0.1,\
                 epochs=25000, batch_size=128,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=False)

#save model
model.save('./model/final_sf.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))

























