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

# ref:[data,name]
path='./airfoil_1600_1aoa_1re/'
data_file='data_cp_fp_1600.pkl'

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

np.random.seed(123)
N= len(name)
I = np.arange(N)
np.random.shuffle(I)
n=1200

#training
up_tr=up[I][:n]
lr_tr=lr[I][:n]
foil_tr=foil[I][:n]
name_tr=name[I][:n]

#test
up_ts=up[I][n:]
lr_ts=lr[I][n:]
foil_ts=foil[I][n:]
name_ts=name[I][n:]



data1=[up_tr,lr_tr,foil_tr,xx,name_tr]
with open(path+'data_1600_tr.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)

data2=[up_ts,lr_ts,foil_ts,xx,name_ts]
with open(path+'data_1600_ts.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)










