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

'''
# simple conv plot

path='./selected_model/case_9_naca_lam_1/'


tmp=[]
with open(path +'0', 'r') as infile:
    data0=infile.readlines()
    for line in data0:            
        if 'val_loss' in line:
            tmp.append(line)
            
with open(path +'1', 'r') as infile:
    data0=infile.readlines()
    for line in data0:            
        if 'val_loss' in line:
            tmp.append(line)            
            
with open(path +'2', 'r') as infile:
    data0=infile.readlines()
    for line in data0:            
        if 'val_loss' in line:
            tmp.append(line)            
            
val_loss=[]
loss=[]

#first element
loss.append('0.2')
val_loss.append('0.21')

for i in range(len(tmp)-1):
    print i
    val_loss.append(tmp[i].split('val_loss:')[1].strip())            
    loss.append(tmp[i].split('- val_loss:')[0].split('loss:')[1].strip())        
'''








            
path='./selected_model/case_11_naca_lam_np_lr1em4/'
data_file='hist.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
h=result[0]
#hist
plt.figure(figsize=(6,5))
plt.plot(range(len(h['loss'])),h['loss'],'r',marker='o', mfc='None',mew=1.5,ms=12,markevery=15,lw=3,label='training_error')
plt.plot(range(len(h['val_loss'])),h['val_loss'],'b',marker='s', mfc='None',mew=1.5,ms=12,markevery=15,lw=3,label='validation_error')
plt.legend(fontsize=20)
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
#plt.xlim([-0.05,1000])
#plt.ylim([-0.2,0.2])    
plt.savefig('./plot_ts/conv_mlp.png', bbox_inches='tight',dpi=200)
plt.show()
            
