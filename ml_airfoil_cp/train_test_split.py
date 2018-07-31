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
data_file='foil_cp_s144_1343.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp=result[0]
out=result[1]
xx=result[2]
name=result[3]
#name=result[4]

inp=np.asarray(inp)
out=np.asarray(out)
name=np.asarray(name)

np.random.seed(123)
N= len(name)
I = np.arange(N)
np.random.shuffle(I)
n=1200

#training
inp_tr=inp[I][:n]
out_tr=out[I][:n]
name_tr=name[I][:n]

#test
inp_ts=inp[I][n:]
out_ts=out[I][n:]
name_ts=name[I][n:]



data1=[inp_tr,out_tr,xx,name_tr]
with open(path+'data_cp_s144_tr.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)

data2=[inp_ts,out_ts,xx,name_ts]
with open(path+'data_cp_s144_ts.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)










