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
# ref:[data,name]
path='./'
data_file='ml_input_output.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
my_inp=result[0]
my_out1=result[1]
my_out2=result[2]
my_out3=result[3]

my_out1=np.asarray(my_out1)
my_out2=np.asarray(my_out2)
my_out3=np.asarray(my_out3)


reno=result[4]
aoa=result[5]
name=result[6]

#CNN-ML
# ---------ML PART:-----------#


## Training sets
xtr1 = my_inp
xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))  

my_out=np.concatenate((my_out1[:,None],my_out2[:,None],my_out3[:,None]),axis=1)
my_out=my_out[:,:,0]


my_error=[]    
#load_model
model_test=load_model('./selected_model/final_cnn.hdf5')  
pred_out=model_test.predict(xtr1)




