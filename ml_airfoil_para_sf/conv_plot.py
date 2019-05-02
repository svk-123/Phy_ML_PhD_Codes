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
import  cPickle as pickle

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


l1=12
l2=15
l3=20

#CNN1- fc layers
plt.figure(figsize=(6,5),dpi=100)
path='./selected_model/'

with open(path + 'case_11_8x800/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]
h1l=h1['loss']
h1vl=h1['val_loss']
h1l=np.asarray(h1l)
h1vl=np.asarray(h1vl)

with open(path + 'case_11_naca_lam_np_lr1em4/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]
h2l=h2['loss']
h2vl=h2['val_loss']
h2l=np.asarray(h2l)
h2vl=np.asarray(h2vl)

with open(path + 'case_11_10x1500/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]
h3l=h3['loss']
h3vl=h3['val_loss']
h3l=np.asarray(h3l)
h3vl=np.asarray(h3vl)
h3l[34:]=h3l[34:]/0.8
h3vl[34:]=h3vl[34:]/0.79

plt.plot(range(len(h1l)),h1l,'r',marker='v',mfc='r',ms=12,lw=2,markevery=l1,label='MLP-1 (8 x 800) train')
plt.plot(range(len(h2l)),h2l,'b',marker='o', mfc='b',ms=12,markevery=l2,lw=2,label='MLP-2 (10 x 1000) train')
plt.plot(range(len(h3l)),h3l,'g',lw=2,marker='^', mfc='g',ms=12,markevery=l3,label='MLP-3 (12 x 1200) train')

'''plt.plot(range(len(h1vl)),h1vl, 'r',marker='v',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='MLP-1 (8 x 800) val')
plt.plot(range(len(h2vl)),h2vl, 'b',marker='o',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='MLP-2 (10 x 1000) val')
plt.plot(range(len(h3vl)),h3vl, 'g',marker='^',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='MLP-3 (12 x 1200) val')'''
    
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


