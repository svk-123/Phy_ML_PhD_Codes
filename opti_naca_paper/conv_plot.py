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


l1=200
l2=250
l3=20

#CNN1- fc layers
plt.figure(figsize=(6,5),dpi=100)
path='./selected_model/'

with open(path + 'turb_3para_st_6x30/hist.pkl', 'rb') as infile:
    result = pickle.load(infile,encoding='bytes')
h1=result[0]
h1l=h1[b'loss']
h1vl=h1[b'val_loss']
h1l=np.asarray(h1l)
h1vl=np.asarray(h1vl)


plt.plot(range(len(h1l)),h1l,'r',marker='v',mfc='r',ms=12,lw=2,markevery=l1,label='Training')
plt.plot(range(len(h1vl)),h1vl,'b',marker='o', mfc='b',ms=12,markevery=l2,lw=2,label='Validation')

    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.25, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
plt.figtext(0.40, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))

#plt.xlim([-50,2000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('mlp_train.tiff', format='tiff', bbox_inches='tight',dpi=300)
plt.show()


