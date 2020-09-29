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
path='./data_file/'
data_file='foil_uiuc.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp=result[0]
out=result[1]
xx=result[2]
name=result[3]

inp=np.asarray(inp)
my_out=np.asarray(out)

xtr1=inp
ttr1=my_out 

xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))  
model_test=load_model('./selected_model/case_1_p8_tanh_uiuc_aug_gan0p5_naca4_v2/model_cnn/model_cnn_1900_0.000010_0.000013.hdf5')  
       
out=model_test.predict([xtr1])
out=out*0.2

#101 to 100 pts
xx=xx[:100]

#calculate error norm
train_l2=[]
train_l1=[]
for k in range(len(out)):    
    
    tmp=my_out[k]-out[k]
    
    train_l2.append( (LA.norm(tmp)/LA.norm(out))*100 )

    tmp2=tmp/out[k]
    train_l1.append(sum(abs(tmp2))/len(out))

print ('train_l2', sum(train_l2)/len(train_l2))

#spread_plot
plt.figure(figsize=(6,5),dpi=100)
plt.plot([-0.2,0.3],[-0.2,0.3],'k',lw=3)
plt.plot(my_out[0],out[0],'r+')
for k in range(len(name)):
    
    plt.plot(my_out[k],out[k],'ro')
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
plt.savefig('ts_tot_1.tiff',format='tiff', bbox_inches='tight',dpi=300)
plt.show()





for k in range(len(train_l2)):
    if (train_l2[k] > 0.2):
        print k
    
        
        plt.figure(figsize=(6,5))
        plt.plot(xx,my_out[k,:],'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=1,label='True')
        plt.plot(xx,out[k,:],'r',lw=3,label='CNN-C5F7')
    
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.2,0.25])
        plt.legend(fontsize=20,frameon=False)
        plt.xlabel('X/c',fontsize=20)
        plt.ylabel('Y',fontsize=20)  
        #plt.axis('off')
        plt.figtext(0.40, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24) 
        plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
        plt.savefig('./plot/ts_%s_%s.png'%(k,name[k]),format='png',bbox_inches='tight',dpi=100)
        plt.show()
    




