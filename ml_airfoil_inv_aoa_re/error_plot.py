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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
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
# ref:[data,name]
path='./'


#train
data_file='train_output_with_l2.pkl'


with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
print result[-1:]

my_out=result[0]
out=result[1]
reno=result[2]
aoa=result[3]
xx=result[4]
name=result[5]
train_l2=result[6]

#error plot
plt.figure(figsize=(6,5),dpi=100)
plt.hist(train_l2[:,0], 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.ylabel('Number of Samples',fontsize=20)
plt.xlim([-0.01,0.15])
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.savefig('train_error.tiff', bbox_inches='tight',dpi=200)
plt.show()

print ('avg l2:',sum(train_l2[:,0])/len(train_l2[:,0]))



#test
data_file='test_output_with_l2.pkl'


with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
print result[-1:]

my_out=result[0]
out=result[1]
reno=result[2]
aoa=result[3]
xx=result[4]
name=result[5]
test_l2=result[6]

#error plot
plt.figure(figsize=(6,5),dpi=100)
plt.hist(test_l2[:,0], 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.ylabel('Number of Samples',fontsize=20)
plt.xlim([-0.01,1])
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.savefig('test_error.tiff', bbox_inches='tight',dpi=200)
plt.show()

print ('avg l2:',sum(test_l2[:,0])/len(test_l2[:,0]))