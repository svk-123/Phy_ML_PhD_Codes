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
folder = './model_piml/'
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
#B=bwd=np.concatenate((Btmp, wdtmp.reshape(len(wdtmp),1)), axis=1)
B=Btmp

bD=np.zeros((l,6))
bD[:,0]=bDtmp[:,0]
bD[:,1]=bDtmp[:,1]
bD[:,2]=bDtmp[:,2]
bD[:,3]=bDtmp[:,4]
bD[:,4]=bDtmp[:,5]
bD[:,5]=bDtmp[:,8]

# std scaler
#L_ss = preprocessing.StandardScaler() 
#L_ss.fit(L)
#L  = L_ss.transform(L)

#bD_sscaler = preprocessing.StandardScaler() 
#bD_sscaler.fit(bD)
#bD = bD_sscaler.transform(bD)


bD1=bD[:,0]
bD2=bD[:,1]
bD3=bD[:,2]
bD4=bD[:,3]
bD5=bD[:,4]
bD6=bD[:,5]

'''
bD1=np.reshape(bD1,(len(bD1),1))
bD2=np.reshape(bD2,(len(bD2),1))
bD3=np.reshape(bD3,(len(bD3),1))
bD4=np.reshape(bD4,(len(bD4),1))
bD5=np.reshape(bD5,(len(bD5),1))
bD6=np.reshape(bD6,(len(bD6),1))
'''

'''
#minMax scaler
B_mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
B_mm.fit(B)
B  = B_mm.transform(B)

#minMax scaler
bD1_mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
bD1_mm.fit(bD1)
bD1  = bD1_mm.transform(bD1)

bD2_mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
bD2_mm.fit(bD2)
bD2  = bD2_mm.transform(bD2)

bD3_mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
bD3_mm.fit(bD3)
bD3  = bD3_mm.transform(bD3)

bD4_mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
bD4_mm.fit(bD4)
bD4  = bD4_mm.transform(bD4)

bD5_mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
bD5_mm.fit(bD5)
bD5  = bD5_mm.transform(bD5)

bD6_mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
bD6_mm.fit(bD6)
bD6  = bD6_mm.transform(bD6)

data=[B_mm,bD1_mm,bD2_mm,bD3_mm,bD4_mm,bD5_mm,bD6_mm]

name='B_bD1_to_bD6'
with open('./scaler/%s.pkl'%name, 'wb') as outfile:
    pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
'''    

'''
name='B_bD1_to_bD6'
with open('./scaler/%s.pkl'%name, 'rb') as infile:
    scaler= pickle.load(infile)
    
B_mm=scaler[0]   
B  = B_mm.transform(B)
 
bD1_mm=scaler[1] 
bD2_mm=scaler[2] 
bD3_mm=scaler[3] 
bD4_mm=scaler[4] 
bD5_mm=scaler[5] 
bD6_mm=scaler[6] 

bD1  = bD1_mm.transform(bD1)
bD2  = bD2_mm.transform(bD2)
bD3  = bD3_mm.transform(bD3)
bD4  = bD4_mm.transform(bD4)
bD5  = bD5_mm.transform(bD5)
bD6  = bD6_mm.transform(bD6)
'''

# ---------ML PART:-----------#
#shuffle data
N= len(L)
I = np.arange(N)
np.random.shuffle(I)
n=10000

## Training sets
#xtr0 = L[I][:n]
xtr0 = B[I][:n]

ttr1 = bD1[I][:n]
ttr2 = bD2[I][:n]
ttr3 = bD3[I][:n]
ttr4 = bD4[I][:n]
ttr5 = bD5[I][:n]
ttr6 = bD6[I][:n]


ttr=np.concatenate((ttr1.reshape(len(ttr1),1),\
                    ttr2.reshape(len(ttr1),1),\
                    ttr3.reshape(len(ttr1),1),\
                    ttr4.reshape(len(ttr1),1),\
                    ttr5.reshape(len(ttr1),1),\
                    ttr6.reshape(len(ttr1),1)), axis=1)      
                                
# Multilayer Perceptron
# create model
aa=Input(shape=(47,))
xx =Dense(30, kernel_initializer='random_normal')(aa)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(30)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(30)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(30)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(30)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(30)(xx)
xx=LeakyReLU(alpha=.1)(xx)


uu =Dense(30)(xx)
uu=LeakyReLU(alpha=.1)(uu)
uu =Dense(30)(uu)
uu=LeakyReLU(alpha=.1)(uu)
uu =Dense(30)(uu)
uu=LeakyReLU(alpha=.1)(uu)
uu =Dense(10)(uu)
g1 =Dense(1, activation='linear')(uu)


#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[g1])

#model = Model(inputs=[aa,t5], outputs=[y5])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=200, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-6, patience=300, verbose=1, mode='auto')

filepath="./model_piml/2_model_piml_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"
filepath_weight="./model_piml/2_weight_model_piml_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=200)
chkpt_weight= ModelCheckpoint(filepath_weight, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=True, mode='auto', period=200)
# Compile model
opt = Adam(lr=2.5e-6)

model.compile(loss= 'mean_squared_error',optimizer= opt)

hist = model.fit( [xtr0], [ttr2], validation_split=0.2,\
                 epochs=100, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,chkpt_weight],verbose=1,shuffle=False)

#hist = model.fit([xtr0,xtr5], [ttr5], validation_split=0.3,\
#                 epochs=10000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1,shuffle=True)
#save model
model.save('./model_piml/2_final_piml.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))



















