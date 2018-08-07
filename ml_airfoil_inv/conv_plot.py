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
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
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


# simple conv plot
'''
path='./'
data_file='hist.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
h=result[0]
#hist
plt.figure(figsize=(6,5),dpi=100)
plt.plot(range(len(h['loss'])),h['loss'],'r',lw=3,label='training_error')
plt.plot(range(len(h['val_loss'])),h['val_loss'],'b',lw=3,label='validation_error')
plt.legend(fontsize=20)
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
#plt.xlim([-0.05,1000])
#plt.ylim([-0.2,0.2])    
plt.savefig('convergence.png', bbox_inches='tight',dpi=100)
plt.show()
'''



#conv case train test split plot
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(6,5),dpi=100)
path='./hyper_selected/'

with open(path + 'casetrainsplit/case_50/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + 'casetrainsplit/case_60/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]

with open(path + 'casetrainsplit/case_70/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]

with open(path + 'casetrainsplit/case_80/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h4=result[0]

#plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'r',lw=4,label='CNN-2 50-50 Training')
#plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'--r',lw=4,label='CNN-2 50-50 Validation')

plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'k',lw=2,markevery=50,label='60%-40% train')
plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'k',marker='o', mfc='grey',ms=10,markevery=80,lw=2,label='70%-30% train')
plt.plot(range(len(h4['loss'][:2000])),h4['loss'][:2000],'k',lw=2,marker='^', mfc='grey',ms=10,markevery=80,label='80%-20% train')

plt.plot(range(len(h2['val_loss'][:2000])),h2['val_loss'][:2000],'--k',markevery=50,lw=2,label='60%-40% val')
plt.plot(range(len(h3['val_loss'][:2000])),h3['val_loss'][:2000],'--k',marker='o', mfc='grey',ms=10, markevery=80,lw=2,label='70%-30% val')
plt.plot(range(len(h4['val_loss'][:2000])),h4['val_loss'][:2000],'--k',marker='^', mfc='grey',ms=10,markevery=80,lw=2,label='80%-20% val')
    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=2, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')

plt.xticks(range(0,2001,500))

#plt.xlim([-10,2000])
#plt.ylim([-0.2,0.2])    
plt.savefig('conv_ttsplit.png', bbox_inches='tight',dpi=200)
plt.show()


#CNN1,2,3
plt.figure(figsize=(6,5),dpi=100)
path='./hyper_selected/'

with open(path + 'case216/case_1/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + 'case216/case_2/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]

with open(path + 'case216/case_3/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]


plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'k',lw=2,markevery=50,label='CNN-1 train')
plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'k',marker='o', mfc='grey',ms=10,markevery=80,lw=2,label='CNN-2 train')
plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'k',lw=2,marker='^', mfc='grey',ms=10,markevery=80,label='CNN-3 train')

plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'--k',markevery=50,lw=2,label='CNN-1 val')
plt.plot(range(len(h2['val_loss'][:2000])),h2['val_loss'][:2000],'--k',marker='o', mfc='grey',ms=10, markevery=80,lw=2,label='CNN-2 val')
plt.plot(range(len(h3['val_loss'][:2000])),h3['val_loss'][:2000],'--k',marker='^', mfc='grey',ms=10,markevery=80,lw=2,label='CNN-3 val')
    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=2, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')

plt.xticks(range(0,2001,500))

#plt.xlim([-10,2000])
#plt.ylim([-0.2,0.2])    
plt.savefig('conv_cnns.png', bbox_inches='tight',dpi=200)
plt.show()



#input size
plt.figure(figsize=(6,5),dpi=100)
path='./hyper_selected/'

with open(path + 'case216/case_2/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + 'case144/case_2/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]

with open(path + 'casesing/case_2/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]


plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'k',lw=2,markevery=50,label='216x216x2 train')
plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'k',marker='o', mfc='grey',ms=10,markevery=80,lw=2,label='144x144x2 train')
plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'k',lw=2,marker='^', mfc='grey',ms=10,markevery=80,label='216x216x1 train')

plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'--k',markevery=50,lw=2,label='216x216x2 val')
plt.plot(range(len(h2['val_loss'][:2000])),h2['val_loss'][:2000],'--k',marker='o', mfc='grey',ms=10, markevery=80,lw=2,label='144x144x2 val')
plt.plot(range(len(h3['val_loss'][:2000])),h3['val_loss'][:2000],'--k',marker='^', mfc='grey',ms=10,markevery=80,lw=2,label='216x216x1 val')
    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.1, 1], ncol=2, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')

plt.xticks(range(0,2001,500))

#plt.xlim([-10,2000])
#plt.ylim([-0.2,0.2])    
plt.savefig('conv_input.png', bbox_inches='tight',dpi=200)
plt.show()


