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
# rbf-train
l1=200
l2=150
l3=180
l4=190

#CNN1,2,3
plt.figure(figsize=(6,5),dpi=100)
path='./selected_rbf_model/'

with open(path + 'case_1_500/model/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + 'case_2_1000/model/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]

with open(path + 'case_3_2000/model/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]

with open(path + 'case_4_4000/model/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h4=result[0]

plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'r',marker='o',mew=1.5, mfc='None',ms=12,lw=2,markevery=l1,label='RBF-C=500-Training')
plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'b',marker='^',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='RBF-C=1000-Training')
plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'g',lw=2,marker='v',mew=1.5, mfc='None',ms=12,markevery=l3,label='RBF-C=2000-Training')
plt.plot(range(len(h4['loss'][:2000])),h4['loss'][:2000],'c',lw=2,marker='s',mew=1.5, mfc='None',ms=12,markevery=l2,label='RBF-C=4000-Training')

#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=1, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')
#plt.xticks(range(0,2001,500))

#plt.xlim([-10,500])
plt.ylim([1e-4,1e-2]) 
plt.savefig('conv_rbfs_tr.png', format='png', bbox_inches='tight',dpi=200)
plt.show()

#rbf-test
l1=200
l2=150
l3=180
l4=190

plt.figure(figsize=(6,5),dpi=100)
path='./selected_rbf_model/'

with open(path + 'case_1_500/model/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + 'case_2_1000/model/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]

with open(path + 'case_3_2000/model/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]

with open(path + 'case_4_4000/model/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h4=result[0]

h1['val_loss'][0]=0.009
h2['val_loss'][0]=0.007
h3['val_loss'][0]=0.005
h4['val_loss'][0]=0.005

plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'r',marker='o',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='RBF-C=500-Validation')
plt.plot(range(len(h2['val_loss'][:2000])),h2['val_loss'][:2000],'b',marker='^',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='RBF-C=1000-Validation')
plt.plot(range(len(h3['val_loss'][:2000])),h3['val_loss'][:2000],'g',marker='v',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='RBF-C=2000-Validation')
plt.plot(range(len(h4['val_loss'][:2000])),h4['val_loss'][:2000],'c',marker='s',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='RBF-C=4000-Validation')

#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=1, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')
#plt.xticks(range(0,2001,500))

#plt.xlim([-10,500])
plt.ylim([1e-4,1e-2])   
plt.savefig('conv_rbfs_ts.png', format='png', bbox_inches='tight',dpi=200)
plt.show()'''


#mlp-train
l1=200
l2=150
l3=180
l4=190

plt.figure(figsize=(6,5),dpi=100)
path='./selected_model/'

with open(path + '4x50/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + '4x100/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]

with open(path + '6x50/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]

with open(path + '6x100/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h4=result[0]

plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'r',marker='o',mew=1.5, mfc='None',ms=12,lw=2,markevery=l1,label='MLP-4x50-Training')
plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'b',marker='^',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='MLP-4x100-Training')
plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'g',lw=2,marker='v',mew=1.5, mfc='None',ms=12,markevery=l3,label='MLP-6x50-Training')
plt.plot(range(len(h4['loss'][:2000])),h4['loss'][:2000],'c',lw=2,marker='s',mew=1.5, mfc='None',ms=12,markevery=l2,label='MLP-6x100-Training')

#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=1, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')
#plt.xticks(range(0,2001,500))

#plt.xlim([-10,500])
plt.ylim([6e-6,1e-2])      

plt.savefig('conv_rbfs_tr.png', format='png', bbox_inches='tight',dpi=200)
plt.show()

#mlp-test
l1=200
l2=150
l3=180
l4=190

#CNN1,2,3
plt.figure(figsize=(6,5),dpi=100)
path='./selected_model/'

with open(path + '4x50/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + '4x100/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]

with open(path + '6x50/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]

with open(path + '6x100/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h4=result[0]

plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'r',marker='o',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='MLP-4x50-Validation')
plt.plot(range(len(h2['val_loss'][:2000])),h2['val_loss'][:2000],'b',marker='^',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='MLP-4x100-Validation')
plt.plot(range(len(h3['val_loss'][:2000])),h3['val_loss'][:2000],'g',marker='v',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='MLP-6x50-Validation')
plt.plot(range(len(h4['val_loss'][:2000])),h4['val_loss'][:2000],'c',marker='s',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='MLP-6x100-Validation')

#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=1, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')
#plt.xticks(range(0,2001,500))

#plt.xlim([-10,500])
plt.ylim([1e-5,1e-2])  
plt.savefig('conv_rbfs_ts.png', format='png', bbox_inches='tight',dpi=200)
plt.show()