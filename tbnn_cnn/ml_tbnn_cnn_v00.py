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

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

import os, shutil
folder = './model_tbnn_cnn_v00/'
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
with open('../tbnn_v1/datafile/to_ml/ml_duct_Re2200_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Ltmp.extend(result[0])
Ttmp.extend(result[1])
bDtmp.extend(result[2])
Btmp.extend(result[9])
bRtmp.extend(result[6])
wdtmp.extend(result[10])

with open('../tbnn_v1/datafile/to_ml/ml_duct_Re2600_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Ltmp.extend(result[0])
Ttmp.extend(result[1])
bDtmp.extend(result[2])
Btmp.extend(result[9])
bRtmp.extend(result[6])
wdtmp.extend(result[10])

with open('../tbnn_v1/datafile/to_ml/ml_duct_Re2900_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Ltmp.extend(result[0])
Ttmp.extend(result[1])
bDtmp.extend(result[2])
Btmp.extend(result[9])
bRtmp.extend(result[6])
wdtmp.extend(result[10])

with open('../tbnn_v1/datafile/to_ml/ml_duct_Re3500_full.pkl', 'rb') as infile:
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

bD1=bD[:,0]
bD2=bD[:,1]
bD3=bD[:,2]
bD4=bD[:,3]
bD5=bD[:,4]
bD6=bD[:,5]

bD1=np.reshape(bD1,(len(bD1),1))
bD2=np.reshape(bD2,(len(bD2),1))
bD3=np.reshape(bD3,(len(bD3),1))
bD4=np.reshape(bD4,(len(bD4),1))
bD5=np.reshape(bD5,(len(bD5),1))
bD6=np.reshape(bD6,(len(bD6),1))


#scaler_start
name='B_bD1_to_bD6'
with open('../tbnn_v1/scaler/%s.pkl'%name, 'rb') as infile:
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

#scaler_end
T=np.zeros((l,10,6))
T[:,:,0]=Ttmp[:,:,0]
T[:,:,1]=Ttmp[:,:,1]
T[:,:,2]=Ttmp[:,:,2]
T[:,:,3]=Ttmp[:,:,4]
T[:,:,4]=Ttmp[:,:,5]
T[:,:,5]=Ttmp[:,:,8]

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




ttr=np.concatenate((ttr1.reshape(len(ttr1),1),\
                    ttr2.reshape(len(ttr1),1),\
                    ttr3.reshape(len(ttr1),1),\
                    ttr4.reshape(len(ttr1),1),\
                    ttr5.reshape(len(ttr1),1),\
                    ttr6.reshape(len(ttr1),1)), axis=1)      

tmp=np.zeros((len(xtr0),2))
xtr0=np.concatenate((xtr0,tmp),axis=1)
xtr0=np.reshape(xtr0,(len(xtr0),7,7))
xtr0=np.reshape(xtr0,(len(xtr0),7,7,1))
         
# Multilayer Perceptron
# create model
# construct model
aa = Input([7,7,1])
bb = ZeroPadding2D(((1,1),(1,1)))(aa)

# 2 3x3 convolutions followed by a max pool
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(bb)
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

# 2 3x3 convolutions followed by a max pool
conv2 = Conv2D(64, (2, 2), activation='relu', padding='same')(pool1)
conv2 = Conv2D(128, (2, 2), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(256, (2, 2), activation='relu', padding='same')(pool2)
#conv3 = Conv2D(512, (2, 2), activation='relu', padding='same')(conv3)

# flatten the 4D array (batch, height, width, depth) into 
# a 2D array (batch, n). Perform a fully connected layer
flat4 = Flatten()(conv3)
#flat5 = Dense(515, activation='relu')(flat4)
flat5 = Dense(256, activation='relu')(flat4)
flat5 = Dense(128, activation='relu')(flat5)
flat5 = Dense(64, activation='relu')(flat5)

# Dropout at 50% on this layer
flat5_dropout = Dropout(0.1)(flat5)

# One more layer to a single value (this will be the predicted drag)
g = Dense(10, activation='linear')(flat5_dropout)

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
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=100, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-6, patience=200, verbose=1, mode='auto')

filepath="./model_tbnn_cnn_v00/model_tbnn_cnn_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"
filepath_weight="./model_tbnn_cnn_v00/weight_model_tbnn_cnn_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=50)

chkpt_weight= ModelCheckpoint(filepath_weight, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=True, mode='auto', period=50)
# Compile model
opt = Adam(lr=2.5e-6)

#scaler
#model.load_weights('./selected_model_tbnn_cnn/wo_scaler/weight_model_tbnn_cnn_1249_0.016_0.014.hdf5')

model.compile(loss= 'mean_squared_error',optimizer= opt)

hist = model.fit([xtr0,xtr1,xtr2,xtr3,xtr4,xtr5,xtr6], [ttr1,ttr2,ttr3,ttr4,ttr5,ttr6], validation_split=0.2,\
                 epochs=5000, batch_size=32,callbacks=[reduce_lr,e_stop,chkpt,chkpt_weight],verbose=1,shuffle=False)

#hist = model.fit([xtr0,xtr5], [ttr5], validation_split=0.3,\
#                 epochs=10000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1,shuffle=True)

#save model
model.save('./model_tbnn_cnn_v00/final_tbnn_cnn.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))





















