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
import pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten,UpSampling2D
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K
from numpy import linalg as LA
import os, shutil

from os import listdir
from os.path import isfile, join, isdir

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
#plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rc('font', family='serif')

#matplotlib.rcParams["font.family"] = "Times"
#matplotlib.rc('font',**{'family':'serif','serif':['Times']})
#matplotlib.rc('text', usetex=True)
# u'LMRoman10'


"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

x1=[40000,60000]
y1=[1.15,1.15]

x2=[50000,50000]
y2=[1.1,1.2]

x3=[50000]
y3=[1.15]

#cpmbined
L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):
    plt.plot(x1,y1,'r',marker='o',mfc='r',ms=16,lw=1,markevery=1)
    plt.plot(x2,y2,'r',marker='o',mfc='r',ms=16,lw=1,markevery=1)
    plt.plot(x3,y3,'g',marker='o',mfc='g',ms=16,lw=1,markevery=1)    
#plt.plot(pinn_relu[:,0],pinn_relu[:,1],'b',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='PINN-Relu')
#plt.plot(nn_tanh[:,0],nn_tanh[:,1],'r',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='NN-Tanh')
#plt.plot(nn_relu[:,0],nn_relu[:,1],'b',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='NN-Relu')

plt.grid('on')
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Re',fontsize=20)
plt.ylabel('$C_L$',fontsize=20)
plt.figtext(0.1, 0.53, 'P1', wrap=True, horizontalalignment='center', fontsize=16)    
plt.figtext(0.5, 0.3, 'P2', wrap=True, horizontalalignment='center', fontsize=16) 
plt.figtext(0.5, 0.53, 'P3', wrap=True, horizontalalignment='center', fontsize=16) 
plt.figtext(0.5, 0.85, 'P4', wrap=True, horizontalalignment='center', fontsize=16) 
plt.figtext(0.85, 0.5, 'P5', wrap=True, horizontalalignment='center', fontsize=16) 

plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))

#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/five_point.tiff', format='tiff', bbox_inches='tight',dpi=300)
plt.show()
