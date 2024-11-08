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

import os, shutil

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

# ref:[data,name]
path='./airfoil_1600_1aoa_1re/'
data_file='cp_foil_1600.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
cp_up=result[0]
cp_lr=result[1]
foil=result[2]
xx=result[3]
name=result[4]


data_file='data_cp_fp_144_1600.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
name1=result[4]
for i in range(len(name)):
    if (name[i]!=name1[i]):
        print 'Not match'


l=25
plt.figure(figsize=(6,5),dpi=100)
    
plt.plot(xx,foil[l][0:35],'g',lw=5,label='true')
plt.plot(xx,foil[l][35:],'g',lw=5)

plt.xlim([-0.05,1.05])
plt.ylim([-0.1,0.1])
#plt.legend(fontsize=16)
#plt.xlabel('X/C',fontsize=16)
#plt.ylabel('Y',fontsize=16)  
#plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.grid()
plt.tight_layout()
plt.savefig('foil1.png',bbox_inches='tight',dpi=100)
plt.show()

'''
l=9
plt.figure(figsize=(6,5),dpi=100)
for k in range(1277,1278):

    
    plt.plot(cp_up[k][:,0],cp_up[k][:,1],'-k',lw=2.0)
    plt.plot(cp_lr[k][:,0],cp_lr[k][:,1],'-k',lw=2.0)
  

#plt.plot(cp_up[l][:,0],cp_up[l][:,1],'g',lw=5,label='true')
#plt.plot(cp_lr[l][:,0],cp_lr[l][:,1],'r',lw=5)

#plt.xlim([-0.05,1.05])
#plt.ylim([-2,1.1])
#plt.legend(fontsize=16)
#plt.xlabel('X/C',fontsize=16)
#plt.ylabel('$C_p$',fontsize=16)  
#plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.grid()
    plt.tight_layout()
    plt.savefig('./plot_out_cp/%s_%s.png'%(k,name[k]),bbox_inches='tight',dpi=100)
    plt.show()
    print k

x=cp_up[k][:,0]
y=cp_up[k][:,1]

fu = scipy.interpolate.interp1d(x,y)
u_yy = fu(x)


plt.plot(x,y,x+0.1,u_yy)'''













