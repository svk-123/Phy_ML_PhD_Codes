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
from numpy import linalg as LA
import os, shutil

import matplotlib
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

'''
path='./'
data_file='hist.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
history=result[0]
#hist
plt.figure(figsize=(6,5),dpi=100)
plt.plot(range(len(history['loss'])),history['loss'],'r',lw=3,label='training_error')
plt.plot(range(len(history['val_loss'])),history['val_loss'],'b',lw=3,label='validation_error')
plt.legend(fontsize=20)
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
#plt.xlim([-0.05,1000])
#plt.ylim([-0.2,0.2])    
plt.savefig('convergence.png', bbox_inches='tight',dpi=100)
plt.show()
'''

#conv plot
plt.figure(figsize=(6,5),dpi=100)
path='./hyperparameter/'
data_file=['hist1.pkl','hist2.pkl','hist3.pkl']

with open(path + 'hist1.pkl', 'rb') as infile:
    result = pickle.load(infile)
history1=result[0]

with open(path + 'hist2.pkl', 'rb') as infile:
    result = pickle.load(infile)
history2=result[0]

with open(path + 'hist_naca.pkl', 'rb') as infile:
    result = pickle.load(infile)
history3=result[0]

'''with open(path + 'hist_sing_144.pkl', 'rb') as infile:
    result = pickle.load(infile)
history3=result[0]'''

plt.plot(range(len(history1['loss'][:2000])),history1['loss'][:2000],'r',lw=4,label='CNN-3 (216 x 216 x2) Training')
plt.plot(range(len(history1['val_loss'][:2000])),history1['val_loss'][:2000],'--r',lw=4,label='CNN-3 (216 x 216 x2) Validation')

plt.plot(range(len(history2['loss'][:2000])),history2['loss'][:2000],'--b',lw=3,label='CNN-2 (216 x 216 x2) Training')
plt.plot(range(len(history2['val_loss'][:2000])),history2['val_loss'][:2000],'--b',lw=3,label='CNN-2 (216 x 216 x2) Validation')


plt.plot(range(len(history3['loss'][:2000])),history3['loss'][:2000],'g',lw=4,label='CNN-3 (216 x 216 x2) Training')
plt.plot(range(len(history3['val_loss'][:2000])),history3['val_loss'][:2000],'--g',lw=4,label='CNN-3 (216 x 216 x2) Validation')

    
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=16)
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')
#plt.xlim([-10,1000])
#plt.ylim([-0.2,0.2])    
plt.savefig('convergence_123.png', bbox_inches='tight',dpi=100)
plt.show()
