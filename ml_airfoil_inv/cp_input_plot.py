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
path='./airfoil_1600_1aoa_1re/'

#single cp
data_file='data_sing_144_1600_tr.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
cp=result[0]
foil=result[1]

out=cp[10]


plt.figure(figsize=(6, 6), dpi=100)
xp, yp = np.meshgrid(range(out.shape[0]), range(out.shape[1]))
cp=plt.imshow(out)
plt.axis('off')
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.00, wspace = 0)
plt.savefig('./plot/input_single.tiff', format='tiff', dpi=200)
plt.show()


# cp upper lower
data_file='data_144_1600_tr.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
cp_up=result[0]
cp_lr=result[1]



out=cp_up[10]
plt.figure(figsize=(6, 6), dpi=100)
xp, yp = np.meshgrid(range(out.shape[0]), range(out.shape[1]))
cp=plt.imshow(out)
plt.axis('off')
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.00, wspace = 0)
plt.savefig('./plot/input_up.tiff', format='tiff', dpi=200)
plt.show()


out=cp_lr[10]
plt.figure(figsize=(6, 6), dpi=100)
xp, yp = np.meshgrid(range(out.shape[0]), range(out.shape[1]))
cp=plt.imshow(out)
plt.axis('off')
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.00, wspace = 0)
plt.savefig('./plot/input_lr.tiff', format='tiff', dpi=200)
plt.show()


