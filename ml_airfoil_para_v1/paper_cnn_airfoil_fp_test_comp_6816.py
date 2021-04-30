#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017
"""
import time
start_time = time.time()


# Python 3.5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os import listdir
from os.path import isfile, join
import sys

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers import merge, Input, dot, add, concatenate
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import cPickle as pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten,UpSampling2D
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K
from numpy import linalg as LA
import os, shutil
from scipy.interpolate import interp1d
 
import matplotlib
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
plt.rc('font', family='serif')

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

# ref:[data,name]
path='./selected_model_p12_paper/data_file_new/'
data_file='foil_uiuc.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp=result[0]
out=result[1]
xx=result[2]
name=result[3]

inp=np.asarray(inp)
out=np.asarray(out)
name=np.asarray(name)

out=out/0.25

np.random.seed(123)
I=range(len(inp))
I=np.asarray(I)
np.random.shuffle(I)

xtr1=inp[I][:1300]
ttr1=out[I][:1300]
name_tr1=name[I][:1300]

xts1=inp[I][1300:]
tts1=out[I][1300:]
name_ts1=name[I][1300:]

I=np.random.randint(0,1300,30)
xtr1=xtr1[I]
ttr1=ttr1[I]
name_tr1=name_tr1[I]

I=np.random.randint(0,125,30)
xts1=xts1[I]
tts1=tts1[I]
name_ts1=name_ts1[I]

xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))  
xts1=np.reshape(xts1,(len(xts1),216,216,1)) 



model_6=load_model('./selected_model_p12_paper/P6_C5F7/model_cnn/final_cnn.hdf5') 
model_8=load_model('./selected_model_p12_paper/P8_C5F7/model_cnn/final_cnn.hdf5')  
model_12=load_model('./selected_model_p12_paper/P12_C5F7/model_cnn/final_cnn.hdf5') 
model_16=load_model('./selected_model_p12_paper/P16_C5F7/model_cnn/model_cnn_1525_0.000004_0.000054.hdf5')  



# with a Sequential model
c6 = model_6.predict([xts1])
c8 = model_8.predict([xts1])
c12 = model_12.predict([xts1])
c16 = model_16.predict([xts1])

xx=xx[:100]
for i in range(20):
    print i
    plt.figure(figsize=(6,5),dpi=100)
    
   
    plt.plot(xx,tts1[i,:]*0.25,'k',marker='o', mfc='None',mew=1.0,ms=8,lw=0,label='True')

    plt.plot(xx,c6[i,:]*0.25,'r',lw=2,label='CNN P-6')
    
    plt.plot(xx,c8[i,:]*0.25,'g',lw=2,label='CNN P-8')
     
    plt.plot(xx,c12[i,:]*0.25,'b',lw=2,label='CNN P-12')
    
    plt.plot(xx,c16[i,:]*0.25,'c',lw=2,label='CNN P-16')
    
    plt.legend(fontsize=14, frameon=False, shadow=False, fancybox=False)
    
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.25,0.25])
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)
    plt.savefig('./plot/ts_%s_%04d.tiff'%(name_tr1[i],i), bbox_inches='tight',dpi=300)
    plt.close()
