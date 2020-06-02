#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

inirially aug foils created using all 1550 airfoils

testing aug foils are removed.. only train agu foils are used to aug the aug foils

i.e "foil_param_216_no_aug_tr" foils are used and ts foils removed.

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
name_ts=result[3]

data_file='foil_param_aug_7.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
    
inp=result[0]
out=result[1]
xx=result[2]
name=result[3]

nn=[]
for i in range(len(name)):
    nn.append(name[i].split('_')[0])

c=0
for i in range(len(name)):
    if nn[i] in name_ts:
        print ('yes')
        c=c+1


#
#for i in range(len(my_out)):
#    print i
#    plt.figure(figsize=(6,5),dpi=100)
#    plt.plot(co,my_out[i,:],'r',lw=2)
#    #plt.plot(xx[::-1],foil_fp[i][:35],'b',label='true')
#    #plt.plot(xx,foil_fp[i][35:],'b',label='true')
#    plt.xlim([-0.05,1.05])
#    plt.ylim([-0.25,0.25])
#    plt.savefig('./to_check/%04d.png'%(i), bbox_inches='tight',dpi=100)
#    plt.close()
#    #plt.show()


