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
from os import listdir
from os.path import isfile, join
import sys

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

from numpy import linalg as LA
import os, shutil

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 


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
data_file='data_clcd.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
val_inp=result[0]
val_out=result[1]
name=result[2]

val_inp=np.asarray(val_inp)
val_out=np.asarray(val_out)

val_inp=np.reshape(val_inp,(len(val_inp),144,144,1))  

model_test=load_model('./selected_model/model_cnn_275_0.000001_0.002375.hdf5')  
out=model_test.predict([val_inp])
out[:,0]=out[:,0] * 1.6
out[:,1]=out[:,1] * 0.2

#plot
plt.figure(figsize=(6, 5), dpi=100)
plt0, =plt.plot([-0.5,2],[-0.5,2],'k')
plt1, =plt.plot(val_out[:,0],out[:,0],'+')  
plt.show()

#plot
plt.figure(figsize=(6, 5), dpi=100)
plt0, =plt.plot([-0.0,0.2],[-0.0,0.2],'k')
plt1, =plt.plot(val_out[:,1],out[:,1],'+')  
plt.show()