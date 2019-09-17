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

path='./result_paper_v2/mp_1_relu/'

#mp1-relu
name=['naca0016','naca0028','naca3416','naca4414','naca4524','naca5024','naca4524','naca5024']
Name=['0016','0028','3416','4414','4524','5024']

##mp2-relu
#name=['naca5014','naca5008','naca3012','naca2030','naca2010','naca1416']
#Name=['5014','5008','3012','2030','2010','1416']


conv=[]

for ii in range(5):
    tmp=np.loadtxt(path + 'conv_%s.dat'%name[ii],skiprows=2)
    conv.append(tmp)

N=90
c=['g','b','r','b','m']
plt.figure(figsize=(6,5),dpi=100)
for ii in range(5):

    plt.plot(conv[ii][:N,0], conv[ii][:N,1],'%s'%c[ii],lw=2,label='Base NACA%s'%Name[ii])
        
plt.legend(loc="upper left", bbox_to_anchor=[0, 1], ncol=2, fontsize=12, \
           frameon=False, shadow=False, fancybox=False,title='')
plt.xlim([-0.05,92])
plt.ylim([-0.05,1.1])
plt.xlabel('Iter',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.savefig(path+'conv.png',bbox_inches='tight',dpi=300)
plt.show()
plt.close()









