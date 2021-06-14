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

#load
# duct - list
flist=['Re2200','Re2600','Re2900']

bD=[]
bR=[]
L=[]
T1= []
T2= []
T3= []
T4= []
T5= []
T6= []
S=[]
R=[]


for ii in range(len(flist)):
   
    # for ref: data1 = [UUDi,bD,L]
    with open('./tbnn_data/data_out/ml_data1_%s.pkl'%flist[ii], 'rb') as infile1:
        result1 = pickle.load(infile1)
    # ref: data2 = [T1m,T2m,T3m,T4m,T5m,T6m]
    with open('./tbnn_data/data_out/ml_data2_%s.pkl'%flist[ii], 'rb') as infile2:
        result2 = pickle.load(infile2)
    # ref: data4 = [S,R]
    with open('./tbnn_data/data_out/ml_data4_%s.pkl'%flist[ii], 'rb') as infile4:
        result4 = pickle.load(infile4)    
    
    bD.extend(result1[1])
    bR.extend(result1[3])
    L.extend(result1[2])
    T1.extend(result2[0])
    T2.extend(result2[1])
    T3.extend(result2[2])
    T4.extend(result2[3])
    T5.extend(result2[4])
    T6.extend(result2[5])
    S.extend(result4[0])
    R.extend(result4[1])


S=np.asarray(S)
R=np.asarray(R)
S=S.reshape(len(S),9)
S[:,3]=R[:,0,1]
S[:,6]=R[:,0,2]
S[:,7]=R[:,1,2]


bD=np.asarray(bD)
bR=np.asarray(bR)

# ---------ML PART:-----------#
#shuffle data
N= len(L)
I = np.arange(N)
np.random.shuffle(I)
n=3000

## Training sets
xtr1 = S
ttr1 = bR



# Multilayer Perceptron
# create model
aa=Input(shape=(9,))
xx =Dense(10,  kernel_initializer='random_normal', activation='relu')(aa)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
xx =Dense(10, activation='relu')(xx)
y =Dense(6, activation='linear')(xx)



#model save dir


#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[y])

#callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, mode='min',verbose=1 ,patience=10, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='val_loss', min_delta=1.0e-6, patience=100, verbose=1, mode='auto')

filepath="./model/model_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=1000)

os.system("rm ./graph/* ")
tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

# Compile model
opt = Adam(lr=2.50e-5,decay=1.0e-9)

model.compile(loss= 'mean_squared_error',optimizer= opt)


hist = model.fit(xtr1,ttr1,validation_split=0.2,\
                 epochs=1000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1)

#save model
model.save('./model/final.hdf5') 


print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))























