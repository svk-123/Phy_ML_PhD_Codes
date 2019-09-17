#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

with new parameters tanh_16_v1:

with new flow prediction network using v1.

@author: vinoth
"""
#based on parameters 

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
from keras.layers import merge, Input, dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

from numpy import linalg as LA
import os, shutil
from skimage import io, viewer,util 
from scipy.optimize import minimize

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

from naca import naca4

path='./result_paper_v2/max_cl_1_tanh/'
name=['naca0012','naca0014','naca2412','naca3310','naca4510']

base=[]
opti=[]

for ii in range(5):
    tmp=np.loadtxt(path + 'base_%s.dat'%name[ii])
    base.append(tmp)
    tmp=np.loadtxt(path + 'final_%s.dat'%name[ii])
    opti.append(tmp)

plt.figure(figsize=(6,5),dpi=100)
for ii in range(5):
    if(ii==0):
        plt.plot(base[ii][:,0],base[ii][:,1],'--k',lw=1,label='Base Shapes')
        plt.plot(opti[ii][:,0],opti[ii][:,1],'g',lw=3,label='Optimized Shapes')
    else:
        plt.plot(base[ii][:,0],base[ii][:,1],'--k',lw=1)
        plt.plot(opti[ii][:,0],opti[ii][:,1],'g',lw=3)    
        
plt.legend(fontsize=14)
plt.xlim([-0.05,1.05])
plt.ylim([-0.25,0.25])
plt.xlabel('X',fontsize=16)
plt.ylabel('Y',fontsize=16)
plt.savefig(path+'combine.png',bbox_inches='tight',dpi=300)
plt.show()
plt.close()









