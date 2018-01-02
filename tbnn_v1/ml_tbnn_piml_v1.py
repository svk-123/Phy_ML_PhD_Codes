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

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

Ltmp=[]
Ttmp=[]
bDtmp=[]
Btmp=[]
bRtmp=[]
wdtmp=[]

# ref:[x,tb,y,coord,k,ep,rans_bij,tkedns,I,B,wd]
with open('./datafile/to_ml/ml_duct_Re2200_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Ltmp.extend(result[0])
Ttmp.extend(result[1])
bDtmp.extend(result[2])
Btmp.extend(result[9])
bRtmp.extend(result[6])
wdtmp.extend(result[10])

with open('./datafile/to_ml/ml_duct_Re2600_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Ltmp.extend(result[0])
Ttmp.extend(result[1])
bDtmp.extend(result[2])
Btmp.extend(result[9])
bRtmp.extend(result[6])
wdtmp.extend(result[10])

with open('./datafile/to_ml/ml_duct_Re2900_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Ltmp.extend(result[0])
Ttmp.extend(result[1])
bDtmp.extend(result[2])
Btmp.extend(result[9])
bRtmp.extend(result[6])
wdtmp.extend(result[10])

with open('./datafile/to_ml/ml_duct_Re3500_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Ltmp.extend(result[0])
Ttmp.extend(result[1])
bDtmp.extend(result[2])
Btmp.extend(result[9])
bRtmp.extend(result[6])
wdtmp.extend(result[10])

bDtmp=np.asarray(bDtmp)
Ltmp=np.asarray(Ltmp)
Ttmp=np.asarray(Ttmp)
Btmp=np.asarray(Btmp)
bRtmp=np.asarray(bRtmp)
wdtmp=np.asarray(wdtmp)

# reduce to 6 components
l=len(Ltmp)

L=Ltmp
B=Btmp

#minMax scaler
B_mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
B_mm.fit(B)
B  = B_mm.transform(B)


bD=np.zeros((l,6))
bD[:,0]=bDtmp[:,0]
bD[:,1]=bDtmp[:,1]
bD[:,2]=bDtmp[:,2]
bD[:,3]=bDtmp[:,4]
bD[:,4]=bDtmp[:,5]
bD[:,5]=bDtmp[:,8]

dbD=np.zeros((l,6))
dbD[:,0]=bDtmp[:,0]-bRtmp[:,0]
dbD[:,1]=bDtmp[:,1]-bRtmp[:,1]
dbD[:,2]=bDtmp[:,2]-bRtmp[:,2]
dbD[:,3]=bDtmp[:,4]-bRtmp[:,4]
dbD[:,4]=bDtmp[:,5]-bRtmp[:,5]
dbD[:,5]=bDtmp[:,8]-bRtmp[:,8]


T=np.zeros((l,10,6))
T[:,:,0]=Ttmp[:,:,0]
T[:,:,1]=Ttmp[:,:,1]
T[:,:,2]=Ttmp[:,:,2]
T[:,:,3]=Ttmp[:,:,4]
T[:,:,4]=Ttmp[:,:,5]
T[:,:,5]=Ttmp[:,:,8]


# std scaler
#L_ss = preprocessing.StandardScaler() 
#L_ss.fit(L)
#L  = L_ss.transform(L)

#bD_sscaler = preprocessing.StandardScaler() 
#bD_sscaler.fit(bD)
#bD = bD_sscaler.transform(bD)

#minMax scaler
#L_mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#bD_mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#L_mm.fit(L)
#bD_mm.fit(bD)
#L  = L_mm.transform(L)
#bD = bD_mm.transform(bD)

bD1=bD[:,0]
bD2=bD[:,1]
bD3=bD[:,2]
bD4=bD[:,3]
bD5=bD[:,4]
bD6=bD[:,5]

dbD1=dbD[:,0]
dbD2=dbD[:,1]
dbD3=dbD[:,2]
dbD4=dbD[:,3]
dbD5=dbD[:,4]
dbD6=dbD[:,5]


T1=T[:,:,0]
T2=T[:,:,1]
T3=T[:,:,2]
T4=T[:,:,3]
T5=T[:,:,4]
T6=T[:,:,5]


# ---------ML PART:-----------#
#shuffle data
N= len(L)
I = np.arange(N)
np.random.shuffle(I)
n=10000

## Training sets
#xtr0 = L[I][:n]
xtr0 = B[I][:n]
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
aa=Input(shape=(47,))
xx =Dense(100, kernel_initializer='random_normal')(aa)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(100)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(100)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(100)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(100)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(100)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(100)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(100)(xx)
xx=LeakyReLU(alpha=.1)(xx)
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

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa,t1,t2,t3,t4,t5,t6], outputs=[y1,y2,y3,y4,y5,y6])

#model = Model(inputs=[aa,t5], outputs=[y5])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, mode='min',verbose=1 ,patience=300, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=500, verbose=1, mode='auto')

filepath ="./model/model_duct_B_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"
filepath_weight ="./model/model_weight_duct_B_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=200)
chkpt_weight= ModelCheckpoint(filepath_weight, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=True, mode='auto', period=200)

# Compile model
opt = Adam(lr=2.5e-6)

model.load_weights("./selected_model/model_weight_duct_B_9999_0.006_0.006.hdf5")
model.compile(loss= 'mean_squared_error',optimizer= opt,loss_weights=[1,1,1,1,1,1])

hist = model.fit([xtr0,xtr1,xtr2,xtr3,xtr4,xtr5,xtr6], [ttr1,ttr2,ttr3,ttr4,ttr5,ttr6], validation_split=0.2,\
                 epochs=10000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,chkpt_weight],verbose=1,shuffle=False)

#hist = model.fit([xtr0,xtr5], [ttr5], validation_split=0.3,\
#                 epochs=10000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1,shuffle=True)

#save model
model.save_weights("./selected_model/model_weight.h5")
model.save('./model/final_duct_B.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))




















