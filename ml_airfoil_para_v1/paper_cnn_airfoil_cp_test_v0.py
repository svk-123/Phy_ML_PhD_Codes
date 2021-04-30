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
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
plt.rc('font', family='serif')

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

# ref:[data,name]
path='./selected_model_p12_paper/data_file_new/'
data_file='foil_uiuc.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp=result[0]
out=result[1]
xx=result[2]
name=result[3]

inp=np.asarray(inp)
out=np.asarray(out)
name=np.asarray(name)

#out=out/0.25

np.random.seed(123)
I=range(len(inp))
I=np.asarray(I)
np.random.shuffle(I)

xtr1=inp[I][:1300]
ttr1=out[I][:1300]
name_tr1=name[I][:1300]

xts1=inp[I][1300:]
tts1=out[I][1300:]
name_ts1=name[I][1300:]

# I=np.random.randint(0,1300,30)
# xtr1=xtr1[I]
# ttr1=ttr1[I]
# name_tr1=name_tr1[I]

# I=np.random.randint(0,125,30)
# xts1=xts1[I]
# tts1=tts1[I]
# name_ts1=name_ts1[I]

xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))  
xts1=np.reshape(xts1,(len(xts1),216,216,1)) 



    
#for training samples
    
model=load_model('./selected_model_p12_paper/P12_C5F7/model_cnn/final_cnn.hdf5') 
pred_out = model.predict([xtr1])
pred_out=pred_out*0.25

'''
#calculate error norm
train_l2=[]
for k in range(len(pred_out)):    
    
    tmp=tts1[k,:]-pred_out[k,:]
    
    train_l2.append( (LA.norm(tmp)/LA.norm(tts1[k,:])))
'''

#spread_plot
plt.figure(figsize=(6,5),dpi=100)
plt.plot([-0.25,0.25],[-0.25,0.25],'k',lw=3)
#plt.plot(my_out[0],out[0],'ro')
for k in range(len(pred_out)):
    
    plt.plot(ttr1[k,:],pred_out[k,:],'+')
plt.legend(fontsize=20)
plt.xlabel('True',fontsize=20)
plt.ylabel('Prediction',fontsize=20)
#lt.xlim([-1.5,1.5])
#plt.ylim([-1.5,1.5])    
plt.savefig('trainn_spread_1.png', bbox_inches='tight',dpi=100)
plt.show()          

#error plot
plt.figure(figsize=(6,5),dpi=100)
plt.hist(train_l2, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.ylabel('Number of Samples',fontsize=20)
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.figtext(0.40, 0.01, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xlim([-0.05,1.3])
#plt.xticks([0,0.5,1.])
plt.savefig('tr_p6.tiff',format='tiff', bbox_inches='tight',dpi=300)
plt.show()





