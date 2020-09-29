#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

@author: vinoth
"""
#based on coordinate without bounds

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
import cPickle as pickle

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


"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

#load model
model_test=load_model('./selected_model/model_cnn_275_0.000001_0.002375.hdf5') 

#import base airfoil
tmp=np.loadtxt('./coord_seligFmt_formatted/naca0015.dat',skiprows=1)

global xx
xx=tmp[:,0].copy()
yy=tmp[:,1].copy()

img_mat=[]
global tar_cl
tar_cl=1.0

def loss(co):
   
    figure=plt.figure(figsize=(2,2))
    plt0, =plt.plot(xx,co,'k',linewidth=0.5,label='true')
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.18,0.18)    
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.savefig('./plot/tmp.eps', format='eps')
    plt.close() 
    img = io.imread('./plot/tmp.eps', as_grey=True)  # load the image as grayscale
    img = util.invert(img)
    
    img=np.asarray(img)
    val_inp=np.reshape(img,(1,144,144,1))  
    out=model_test.predict([val_inp])
    pred_cl=out[0,0]*1.6

    return (tar_cl-pred_cl)
   
    
print('Starting loss = {}'.format(loss(yy)))

res = minimize(loss, x0=yy, method = 'L-BFGS-B', bounds=None, \
               options={'disp': True, 'maxcor': 100, 'ftol': 1 * np.finfo(float).eps, \
                                 'eps': 5e-3, 'maxfun': 100, \
                                 'maxiter': 100, 'maxls': 100})
print('Ending loss = {}'.format(loss(res.x)))

#compare
plt.plot(xx,yy,xx,res.x)














