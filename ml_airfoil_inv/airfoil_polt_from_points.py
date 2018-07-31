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
path='./airfoil_1600_1aoa_1re/naca'
indir=path

fname = [f for f in listdir(indir) if isfile(join(indir, f))]


plt.figure(figsize=(6,5),dpi=100)
for i in range(len(fname)):
    
    
    cd=np.loadtxt(path+'/%s'%fname[i],skiprows=1)
    cd1=np.loadtxt('./airfoil_1600_1aoa_1re/naca131/%s'%fname[i],skiprows=1)    

    plt.plot(cd[:,0],cd[:,1],'-k',lw=3.0)
    plt.plot(cd1[:,0],cd1[:,1],'-r',lw=2.0)  
    
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-0.2,0.2])
    plt.grid()
    plt.tight_layout()
    plt.savefig('./naca_out/%s_%s.png'%(i,fname[i]),bbox_inches='tight',dpi=100)
    plt.show()









