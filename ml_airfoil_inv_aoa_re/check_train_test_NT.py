#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 16:53:31 2019

@author: vino
"""

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

# check whether training and testing has no overlap
train_foil=[]
for ii in [1,2,3,4,5,6,7,8,9,10]:
    print ii
#for ii in [9]:         
    # ref:[data,name]
    path='./foil_all_re_aoa/data_files_train_test_NT/'
    data_file='data_re_aoa_fp_NT_tr_%s.pkl'%ii
    
    with open(path + data_file, 'rb') as infile:
        result = pickle.load(infile)
    val6=result[6]
    val6=np.asarray(val6)    
            
    unique, counts = np.unique(val6, return_counts=True)
    train_foil.extend(unique)

test_foil=[]
for ii in [1,2,3,4,5,6,7,8,9,10]:
    print ii
#for ii in [9]:         
    # ref:[data,name]
    path='./foil_all_re_aoa/data_files_train_test_NT/'
    data_file='data_re_aoa_fp_NT_ts_%s.pkl'%ii
    
    with open(path + data_file, 'rb') as infile:
        result = pickle.load(infile)
    val6=result[6]
    val6=np.asarray(val6)    
            
    unique, counts = np.unique(val6, return_counts=True)
    test_foil.extend(unique)        
    
train_foil=np.asarray(train_foil)
test_foil=np.asarray(test_foil)

tmp=[]
for i in range(len(test_foil)):
    tmp.append(np.argwhere(test_foil[i]==train_foil))



