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
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
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

xx1=[]
for i in range(10):
    tmp=np.loadtxt('./coord_shifted_scaled/s6063_%s.dat'%i,skiprows=1)
    xx1.append(tmp)


plt.figure(figsize=(6,5),dpi=100)


for i in range(10):
    print i
    plt.plot(xx1[i][:,0],xx1[i][:,1],'g',lw=0.5)
    
plt.plot(xx1[6][:,0],xx1[6][:,1],'g',lw=1,label='Scaled')
plt.plot(xx1[9][:,0],xx1[9][:,1],'k',lw=3,label='Original')  
#    plt.plot(xx[::-1],ttr1[i,:35],'k',marker='o', mfc='None',mew=1.0,ms=8,lw=0,label='True')
#    plt.plot(xx,ttr1[i,35:],'k',marker='o', mfc='None',mew=1.0,ms=8,lw=0)
    
plt.legend(fontsize=18, frameon=False, shadow=False, fancybox=False)
    
plt.xlim([-0.05,1.05])
plt.ylim([-0.06,0.1])
plt.xlabel('X',fontsize=16)
plt.ylabel('Y',fontsize=16)
plt.savefig('./plot/s6063.tiff', bbox_inches='tight',dpi=300)
plt.show()
plt.close()
