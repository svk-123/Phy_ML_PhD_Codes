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
data_file='foil_param_naca4_opti.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp=result[0]
out=result[1]
xx=result[2]
name=result[3]

inp=np.asarray(inp)
my_out=np.asarray(out)

xtr1=inp
ttr1=my_out 

xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))  
model=load_model('./selected_model/case_aug_tanh_8/model_cnn_1500_0.000016_0.000026.hdf5') 

del inp
del result

# with a Sequential model
get_out_1c= K.function([model.layers[0].input], [model.layers[16].output])
c1 = get_out_1c([xtr1])[0]

mm=[]
for i in range(8):
    mm.append([c1[:,i].max(),c1[:,i].min()])
mm=np.asarray(mm)
    
mm_scale=[]    
for i in range(8):    
    mm_scale.append(max(abs(mm[i])))
mm_scale=np.asarray(mm_scale)

c1_scaled=c1.copy()
for i in range(8): 
    c1_scaled[:,i]=c1_scaled[:,i]/mm_scale[i]
    
#info='[para_scaled,name,para(unscaled),mm_scaler,info]'    
#data2=[c1_scaled,name,c1,mm_scale,info]
#with open(path+'param_naca4_tanh_8_v1.pkl', 'wb') as outfile:
#    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)

