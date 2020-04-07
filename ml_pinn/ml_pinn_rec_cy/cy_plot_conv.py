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
import pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten,UpSampling2D
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K
from numpy import linalg as LA
import os, shutil

from os import listdir
from os.path import isfile, join, isdir

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


path='./tf_model'
tmp=[f for f in listdir(path) if isdir(join(path, f))]
case_dir=np.asarray(tmp)
case_dir.sort()

mylabel=['PINN_2222_3S_around_80','PINN_2222_3S_wake_80','PINN_2222_4S_around_80','PINN_5555_4S_around_80']

data=[]
for i in range(4):
    with open('./tf_model/%s/tf_model/conv.dat'%case_dir[i], 'r') as infile:
        data0=infile.readlines()
    tmp=[]    
    for j in range(1,50000):
        if 'Reduce' not in data0[j]:
            tmp.append(np.asarray(data0[j].split(',')))
    tmp=np.asarray(tmp)
    tmp=tmp.astype(np.float)        
    data.append(tmp)
    






l1=200
l2=250
l3=20
c=['g','b','y','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown'] 
mylabel=['PINN_2222_3S_around_80','PINN_2222_3S_wake_80','PINN_2222_4S_around_80','PINN_5555_4S_around_80']

L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(4):
    plt.plot(data[i][:L,0],data[i][:L,2]+data[i][:L,3],'%s'%c[i+1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='%s'%mylabel[i])
#plt.plot(pinn_relu[:,0],pinn_relu[:,1],'b',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='PINN-Relu')
#plt.plot(nn_tanh[:,0],nn_tanh[:,1],'r',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='NN-Tanh')
#plt.plot(nn_relu[:,0],nn_relu[:,1],'b',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='NN-Relu')


plt.legend(loc="upper left", bbox_to_anchor=[1, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
#plt.figtext(0.40, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))

#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/loss_mse.png', format='png', bbox_inches='tight',dpi=300)
plt.show()


#residual
L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(4):
    plt.plot(data[i][:L,0],data[i][:L,4],'%s'%c[i+1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='%s'%mylabel[i])
#plt.plot(pinn_relu[:,0],pinn_relu[:,1],'b',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='PINN-Relu')
#plt.plot(nn_tanh[:,0],nn_tanh[:,1],'r',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='NN-Tanh')
#plt.plot(nn_relu[:,0],nn_relu[:,1],'b',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='NN-Relu')


plt.legend(loc="upper left", bbox_to_anchor=[1, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('Gov. Residual',fontsize=20)
plt.yscale('log')
#plt.figtext(0.40, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))

#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/loss_gov.png', format='png', bbox_inches='tight',dpi=300)
plt.show()


#cpmbined
L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):
    plt.plot(data[i][:L,0],data[i][:L,4],'r',marker='None',mfc='r',ms=12,lw=3,markevery=l1,label='MSE')
    plt.plot(data[i][:L,0],data[i][:L,2]+data[i][:L,3],'g',marker='None',mfc='r',ms=12,lw=3,markevery=l1,label='Gov. Eq. Res.')
#plt.plot(pinn_relu[:,0],pinn_relu[:,1],'b',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='PINN-Relu')
#plt.plot(nn_tanh[:,0],nn_tanh[:,1],'r',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='NN-Tanh')
#plt.plot(nn_relu[:,0],nn_relu[:,1],'b',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='NN-Relu')


plt.legend(loc="upper left", bbox_to_anchor=[1, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('Error',fontsize=20)
plt.yscale('log')
#plt.figtext(0.40, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))

#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/loss_ppt.png', format='png', bbox_inches='tight',dpi=300)
plt.show()
