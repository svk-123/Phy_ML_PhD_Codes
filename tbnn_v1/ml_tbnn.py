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


import os, shutil
folder = './model/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)


#load
# duct - list
flist=['Re2200','Re2600','Re2900']

bD=[]
L=[]
T1= []
T2= []
T3= []
T4= []
T5= []
T6= []



for ii in range(len(flist)):
   
    # for ref: data1 = [UUDi,bD,L]
    with open('./tbnn_data/data_out/ml_data1_%s.pkl'%flist[ii], 'rb') as infile1:
        result1 = pickle.load(infile1)
    # ref: data2 = [T1m,T2m,T3m,T4m,T5m,T6m]
    with open('./tbnn_data/data_out/ml_data2_%s.pkl'%flist[ii], 'rb') as infile2:
        result2 = pickle.load(infile2)
    
    
    bD.extend(result1[1])
    L.extend(result1[2])
    T1.extend(result2[0])
    T2.extend(result2[1])
    T3.extend(result2[2])
    T4.extend(result2[3])
    T5.extend(result2[4])
    T6.extend(result2[5])
  
    
bD=np.asarray(bD)
L=np.asarray(L)
T=[T1,T2,T3,T4,T5,T6]
T=np.asarray(T)

bD1=bD[:,0]
bD2=bD[:,1]
bD3=bD[:,2]
bD4=bD[:,3]
bD5=bD[:,4]
bD6=bD[:,5]

T1=T[0,:,:]
T2=T[1,:,:]
T3=T[2,:,:]
T4=T[3,:,:]
T5=T[4,:,:]
T6=T[5,:,:]

# ---------ML PART:-----------#
#shuffle data
N= len(L)
I = np.arange(N)
np.random.seed(12345)
np.random.shuffle(I)
n=5000

## Training sets
xtr0 = L[I][:n]
xtr1 = T1[I][:n]
xtr2 = T2[I][:n]
xtr3 = T3[I][:n]
xtr4 = T4[I][:n]
xtr5 = T5[I][:n]
xtr6 = T6[I][:n]

ttr1 = bD1[I][:n]
ttr2 = bD2[I][:n]
ttr3 = bD3[I][:n]
ttr4 = bD4[I][:n]
ttr5 = bD5[I][:n]
ttr6 = bD6[I][:n]

# Multilayer Perceptron
# create model
aa=Input(shape=(5,))
xx =Dense(50,  kernel_initializer='random_normal', activation='relu')(aa)
xx =Dense(50, activation='relu')(xx)
xx =Dense(50, activation='relu')(xx)
xx =Dense(50, activation='relu')(xx)
xx =Dense(50, activation='relu')(xx)
xx =Dense(50, activation='relu')(xx)
xx =Dense(50, activation='relu')(xx)
xx =Dense(50, activation='relu')(xx)
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
#model = Model(inputs=[aa,t1,t2,t3,t4,t5,t6], outputs=[y1,y2,y3,y4,y5,y6])
model = Model(inputs=[aa,t1], outputs=[y1])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, mode='min',verbose=1 ,patience=10, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='val_loss', min_delta=1.0e-6, patience=100, verbose=1, mode='auto')

filepath="./model/model_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=1000)

os.system("rm ./graph/* ")
tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

# Compile model
opt = Adam(lr=2.50e-6,decay=1.0e-9)

model.compile(loss= 'mean_squared_error',optimizer= opt)


#hist = model.fit([xtr0,xtr1,xtr2,xtr3,xtr4,xtr5,xtr6], [ttr1,ttr2,ttr3,ttr4,ttr5,ttr6], validation_split=0.2,\
#                 epochs=10000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1,shuffle=False)

hist = model.fit([xtr0,xtr1], [ttr1], validation_split=0.3,\
                 epochs=10000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1,shuffle=False)
#save model
model.save('./model/final.hdf5') 


print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))























