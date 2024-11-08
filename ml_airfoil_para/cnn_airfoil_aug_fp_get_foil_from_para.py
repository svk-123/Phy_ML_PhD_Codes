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
data_file='param_aug_216_tanh_16_v1.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)
para=result1[2]
name=result1[1]
mm_scaler=result1[3]
para=np.asarray(para)

data_file='foil_param_216_no_aug.pkl'
with open(path + data_file, 'rb') as infile:
    result2 = pickle.load(infile)
foil_fp=result2[1]
xx=result2[2]

del result2

#model=load_model('./selected_model/case_5c/model_cnn_2450_0.000021_0.000032.hdf5') 
model=load_model('./selected_model/case_aug_tanh_16/model_cnn_1125_0.000004_0.000009.hdf5') 

# with a Sequential model
print model.layers[17].input
print model.layers[21].output

mm=[]
for i in range(16):
    mm.append([para[:,i].min(),para[:,i].max()])

mm=np.asarray(mm)

np.random.seed(124254)

#random generated parameters to check whetrer airfoil is smooth:
new_para=np.zeros((10,16))
for j in range(10):
    for i in range(16):
        new_para[j,i]=random.uniform(mm[i,0],mm[i,1])


get_out_1c= K.function([model.layers[17].input],
                                  [model.layers[21].output])
c1 = get_out_1c([new_para])[0]

for i in range(len(c1)):
    print i
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx[::-1],c1[i,:35]*0.25,'r',label='true')
    plt.plot(xx,c1[i,35:]*0.25,'r',label='true')
    #plt.plot(xx[::-1],foil_fp[i][:35],'b',label='true')
    #plt.plot(xx,foil_fp[i][35:],'b',label='true')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.25,0.25])
    plt.savefig('./plot/ts_%04d.png'%(i), bbox_inches='tight',dpi=100)
    plt.close()
    #plt.show()


