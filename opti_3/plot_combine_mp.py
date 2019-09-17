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

path='./result_paper_v2/mp_1_tanh/'

##for mp-1-relu
#name=['naca0016','naca0028','naca0222','naca3416','naca4414','naca4524','naca5024']

#for mp-1-tanh
name=['naca0016','naca3416','naca4414','naca4524']

##for mp-2-relu
#name=['naca5014','naca5008','naca3012','naca2030','naca2010','naca1416']


#name=foil
c=['b','b','y','c','r','m']
base=[]
opti=[]

for ii in range(4):
    tmp=np.loadtxt(path + 'base_%s.dat'%name[ii])
    base.append(tmp)
    tmp=np.loadtxt(path + 'final_%s.dat'%name[ii])
    opti.append(tmp)

plt.figure(figsize=(6,5),dpi=100)
for ii in [0,1,2,3]:
    if(ii==0):
        #plt.plot(base[ii][:,0],base[ii][:,1],'--k',lw=1,label='Base Shapes')
        plt.plot(opti[ii][:,0],opti[ii][:,1],'g',lw=2,label='Optimized Shape-%d'%ii)
    else:
        #plt.plot(base[ii][:,0],base[ii][:,1],'--k',lw=1)
        plt.plot(opti[ii][:,0],opti[ii][:,1],'-%s'%c[ii],lw=2,label='Optimized Shape-%d'%ii)    
        
        
plt.legend(loc="upper left", bbox_to_anchor=[1, 1], ncol=1, fontsize=14, \
           frameon=False, shadow=False, fancybox=False,title='')
plt.xlim([-0.05,1.05])
plt.ylim([-0.25,0.25])
plt.xlabel('X',fontsize=16)
plt.ylabel('Y',fontsize=16)
plt.savefig(path+'combine_to_report.png',bbox_inches='tight',dpi=300)
plt.show()
plt.close()









