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
data_file='.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp_up=result[0]
inp_lr=result[1]
my_out=result[2]
xx=result[3]
name=result[4]

inp_up=np.asarray(inp_up)
inp_lr=np.asarray(inp_lr)
my_out=np.asarray(my_out)

xtr1=np.concatenate((inp_up[:,:,:,None],inp_lr[:,:,:,None]),axis=3) 
ttr1=my_out 


path='./airfoil_1600_1aoa_1re/'
data_file='cp_foil_1600.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
cp_up=result[0]
cp_lr=result[1]


model_test=load_model('./selected_model/dist/model_enc_cnn_225_0.001_0.001.hdf5')  
       
out=model_test.predict([xtr1])
out=out*0.18

for k in range(1):

    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx,my_out[k][0:35],'ro',label='true')
    plt.plot(xx,my_out[k][35:],'ro')
    plt.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
    plt.plot(xx,out[k][35:],'b',lw=2)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.2,0.2])
    plt.legend(fontsize=16)
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)  
    #plt.axis('off')
    plt.tight_layout()
    plt.savefig('./plot_out_dist_ts/ts_%s_%s.png'%(k,name[k]))
    plt.show()
    
    
    '''fig = plt.figure(figsize=(8, 4),dpi=100)
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(xx,my_out[k][0:35],'ro',label='true')
    ax1.plot(xx,my_out[k][35:],'ro')
    ax1.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
    ax1.plot(xx,out[k][35:],'b',lw=2)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.2,0.2])
    #plt.legend(fontsize=16)
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)  
    #plt.axis('off')
    #plt.tight_layout()
 
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(cp_up[k][:,0],cp_up[k][:,1],'k')
    ax2.plot(cp_lr[k][:,0],cp_lr[k][:,1],'k')
    plt.xlim([-0.05,1.05])
    plt.ylim([-2.5,1])
    plt.legend(fontsize=16)
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)  
    #plt.axis('off')
    #plt.tight_layout()
    
    
    plt.savefig('./plot_out/ts_%s_%s.png'%(k,name[k]))
    plt.show()    '''




#calculate error norm
train_l2=[]
train_l1=[]
for k in range(len(name)):    
    
    tmp=my_out[k]-out[k]
    
    train_l2.append( (LA.norm(tmp)/LA.norm(out))*100 )

    tmp2=tmp/out[k]
    train_l1.append(sum(abs(tmp2))/len(out))

'''
#spread_plot
plt.figure(figsize=(6,5),dpi=100)
plt.plot([-1,1],[-1,1],'k',lw=3,label='true')
plt.plot(my_out[0],out[0],'ro',label='prediction')
for k in range(len(name)):
    
    plt.plot(my_out[k],out[k],'ro')
plt.legend(fontsize=16)
plt.xlabel('True',fontsize=16)
plt.ylabel('Prediction',fontsize=16)
plt.xlim([-0.20,0.20])
plt.ylim([-0.20,0.20])    
plt.savefig('./plot_out/test_spread.png')
plt.show()          

#error plot
plt.figure(figsize=(6,5),dpi=100)
plt.hist(train_l2, 20,histtype='bar', stacked=True)
plt.xlabel('L2 relative error',fontsize=16)
plt.ylabel('number of Samples',fontsize=16)
plt.savefig('./plot_out/test_error.png')
plt.show()

path='./selected_model/case_3_fp'
data_file='/hist.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
history=result[0]
#hist
plt.figure(figsize=(6,5),dpi=100)
plt.plot(range(len(history['loss'])),history['loss'],'r',lw=2,label='training_error')
plt.plot(range(len(history['val_loss'])),history['val_loss'],'b',lw=2,label='validation_error')
plt.legend(fontsize=16)
plt.xlabel('Iteration',fontsize=16)
plt.ylabel('MSE',fontsize=16)
#plt.xlim([-0.05,1.05])
#plt.ylim([-0.2,0.2])    
plt.savefig('./plot_out/convergence.png')
plt.show()
'''


