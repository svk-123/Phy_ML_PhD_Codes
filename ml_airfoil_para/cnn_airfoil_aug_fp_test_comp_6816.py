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
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
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
path='./data_file/'
data_file='foil_param_216_no_aug_ts.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp=result[0]
out=result[1]
xx=result[2]
name=result[3]

inp=np.asarray(inp)
my_out=np.asarray(out)

xtr1=inp
ttr1=my_out 

xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))  

np.random.seed(1234534)
I=np.random.randint(0,329,100)

xtr1=xtr1[I]
ttr1=ttr1[I]

model_6=load_model('./selected_model/case_aug_tanh_6/model_cnn_2100_0.000030_0.000043.hdf5') 
model_8=load_model('./selected_model/case_aug_tanh_8/model_cnn_1500_0.000016_0.000026.hdf5') 
model_16=load_model('./selected_model/case_aug_tanh_16/model_cnn_1125_0.000004_0.000009.hdf5') 

del inp
del result

# with a Sequential model
c6 = model_6.predict([xtr1])
c8 = model_8.predict([xtr1])
c16 = model_16.predict([xtr1])

for i in range(len(c6)):
    print i
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx[::-1],c6[i,:35]*0.25,'r',lw=2,label='p-6')
    plt.plot(xx,c6[i,35:]*0.25,'r',lw=2)
    
    plt.plot(xx[::-1],c8[i,:35]*0.25,'g',lw=2,label='p-8')
    plt.plot(xx,c8[i,35:]*0.25,'g',lw=2)    
    
    plt.plot(xx[::-1],c16[i,:35]*0.25,'b',lw=2,label='p-16')
    plt.plot(xx,c16[i,35:]*0.25,'b',lw=2)
    
    plt.plot(xx[::-1],ttr1[i,:35],'gray',marker='o', mfc='None',mew=1.0,ms=8,lw=0,label='true')
    plt.plot(xx,ttr1[i,35:],'gray',marker='o', mfc='None',mew=1.0,ms=8,lw=0)
    
    plt.legend()
    
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.25,0.25])
    
    plt.savefig('./plot/ts_%04d.png'%(i), bbox_inches='tight',dpi=100)
    plt.close()
