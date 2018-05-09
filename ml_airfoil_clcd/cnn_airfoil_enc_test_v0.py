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
path='./naca456'
data_file='/data_airfoil_inverse.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
coord=result

## Training sets
xtr1 = coord[0]
xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))         
ttr1=xtr1
# print dataset values
print('xtr shape:', xtr1.shape)
print('ttr shape:', ttr1.shape)



model_test=load_model('./model_cnn/model_enc_cnn_50_0.006_0.006.hdf5')  


out=model_test.predict([xtr1])

for i in range(10):
    plt.subplot(2, 1, 1)
    plt.imshow(xtr1[i,:,:,0])
    plt.gray()
    plt.show()
        
    plt.subplot(1,2,2)
    plt.imshow(out[i,:,:,0])
    plt.gray()
    plt.show()




