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
path='./data_file/'
#data_file='foil_param_CST_1.pkl'
data_file='foil_param_216_no_aug.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp=result[0]
my_out=result[1]
xx=result[2]
name=result[3]

inp=np.asarray(inp)
my_out=np.asarray(my_out)

xtr1=inp
ttr1=my_out 

xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))  
model_test=load_model('./selected_model/case_CST_tanh_16_v0_more_data/model_cnn_800_0.000001_0.000003.hdf5')  
       
out=model_test.predict([xtr1])
out=out*0.25



#[0,9]
for k in range(1529):
    print k
    yy1=out[k][0:35]
    yy2=out[k][35:]
    yy=np.concatenate((yy1[:,None],yy2[1:,None]))
    
    plt.figure(figsize=(6,5))
    plt.plot(xx[::-1],my_out[k][0:35],'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=1,label='True')
    plt.plot(xx,my_out[k][35:],'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=1)
    
    plt.plot(xx[::-1],out[k][0:35],'r',lw=3,label='CNN')
    plt.plot(xx,out[k][35:],'r',lw=3)
    
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.25,0.25])
    plt.legend(fontsize=20)
    plt.xlabel('X/c',fontsize=20)
    plt.ylabel('Y',fontsize=20)  
    #plt.axis('off')
    plt.figtext(0.40, 0.01, '(b)', wrap=True, horizontalalignment='center', fontsize=24) 
    plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
    plt.savefig('./plot/ts_%s_%s.png'%(k,name[k]),format='png',bbox_inches='tight',dpi=100)
    plt.show()
    



