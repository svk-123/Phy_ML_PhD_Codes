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
import random
import os, shutil

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

'''# ref:[data,name]
path='./airfoil_1600_1aoa_1re/'
data_file='data_cp_fp_144_1600.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
up=result[0]
lr=result[1]
foil=result[2]
xx=result[3]
name=result[4]

up=np.asarray(up)
lr=np.asarray(lr)
foil=np.asarray(foil)
name=np.asarray(name)

tmp=np.loadtxt(path+'best_144.dat')
tmp=tmp.astype(int)

up=up[tmp]
lr=lr[tmp]
foil=foil[tmp]
name=name[tmp]


data1=[up,lr,foil,xx,name]
with open(path+'best_cp_fp_144_1100.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)'''



# ref:[data,name]
path='./data_file/'
data_file='foil_param.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
    
val1=result[0]
val2=result[1]
xx=result[2]
val3=result[3]
info=result[4]


val1=np.asarray(val1)
val2=np.asarray(val2)
val3=np.asarray(val3)


#np.random.seed(123)
np.random.seed(154328)
N= len(val1)
I = np.arange(N)
np.random.shuffle(I)
n=1200

#training
tr_val1=val1[I][:n]
tr_val2=val2[I][:n]
tr_val3=val3[I][:n]


#test
ts_val1=val1[I][n:]
ts_val2=val2[I][n:]
ts_val3=val3[I][n:]



'''data1=[tr_val1,tr_val2,xx,tr_val3,info]
with open(path+'foil_param_tr.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)

data2=[ts_val1,ts_val2,xx,ts_val3,info]
with open(path+'foil_param_ts.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)'''








