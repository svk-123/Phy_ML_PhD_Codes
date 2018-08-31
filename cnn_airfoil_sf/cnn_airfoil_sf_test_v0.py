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

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

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
path='./airfoil_data/'
data_file='foil_aoa_inout.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
print result[-1:]    

st=0
end=100

my_inp=result[0][st:end]
my_out=result[1][st:end]
myco=result[4][st:end]
mybor=result[5][st:end]
myins=result[6][st:end]
name=result[7][st:end]

my_inp=np.asarray(my_inp)
my_out=np.asarray(my_out)

xtr1=np.reshape(my_inp,(len(my_inp),288,216,1))  
ttr1=my_out  

model_test=load_model('./selected_model/p/p_model_cnn_250_0.000176_0.000174.hdf5')  
out=model_test.predict([xtr1])

def plot(zp1,zp2,nc,name):
    xp, yp = np.meshgrid(np.linspace(-1,2,216), np.linspace(1,-1,288))
    
    plt.figure(figsize=(16, 6))
    
    plt.subplot(121)
    plt.contourf(xp,yp,zp1,nc,cmap=cm.jet)
    plt.colorbar()
    plt.title('CFD-%s'%name)
    
    plt.subplot(122)
    plt.contourf(xp,yp,zp2,nc,cmap=cm.jet)
    plt.colorbar()
    #plt.yticks([])
    #plt.yaxis('off')
    plt.title('Prediction-%s'%name)  
    #plt.xlim([-0.1,1.1])
    #plt.ylim([-0.1,0.1])
    plt.savefig('./plotc/%s.png'%(name), format='png',dpi=200)
    plt.show()






for k in range(10):
        
    p1=my_out[k].copy()
    p2=out[k,:,:,0].copy()
    
    xb=mybor[k][:,0]
    yb=mybor[k][:,1]
    
    xi=np.asarray(myins[k])[:,0]
    yi=np.asarray(myins[k])[:,1]    

    p1[xb,yb]= np.nan
    p1[yi,xi]= np.nan
    
    p2[xb,yb]= np.nan
    p2[yi,xi]= np.nan
        
    plot(p1,p2,20,'ts_%s'%name[k])
        
    #plot(p1-p2,20,'%s-e'%name[k])

