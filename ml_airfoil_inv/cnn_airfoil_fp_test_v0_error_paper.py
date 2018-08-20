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
path='./airfoil_1600_1aoa_1re/'
data_file='data_144_1600_tr.pkl'

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


model_test=load_model('./hyper_selected/case144/case_3/final_enc_cnn.hdf5')  
       
out=model_test.predict([xtr1])
out=out*0.18

#calculate error norm
train_l2=[]
train_l1=[]
for k in range(len(name)):    
    
    tmp=my_out[k]-out[k]
    
    train_l2.append( (LA.norm(tmp)/LA.norm(out))*100 )

    tmp2=tmp/out[k]
    train_l1.append(sum(abs(tmp2))/len(out))


##spread_plot
#plt.figure(figsize=(6,5),dpi=100)
#plt.plot([-1,1],[-1,1],'k',lw=3)
#for k in range(len(name)):
#    
#    plt.plot(my_out[k],out[k],'+',c='grey')
#    
#plt.legend(fontsize=20)
#plt.xlabel('True',fontsize=20)
#plt.ylabel('Prediction',fontsize=20)
#plt.xlim([-0.20,0.20])
#plt.ylim([-0.20,0.20])    
#plt.savefig('tr_spread.eps', format='eps', bbox_inches='tight',dpi=100)
#plt.show()          

#error plot
plt.figure(figsize=(6,5),dpi=100)
plt.hist(train_l2, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.ylabel('Number of Samples',fontsize=20)
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.savefig('tr_error_144_cnn3.eps',format='eps', bbox_inches='tight',dpi=100)
plt.show()


#get foil name with lease error
num=np.asarray(range(len(train_l2)))
train_l2=np.asarray(train_l2)
tmp=np.concatenate((train_l2[:,None],num[:,None]),axis=1)
tmp = tmp[tmp[:,0].argsort()]
fp=open('ts_30.dat','w+')
for i in range(30):
    fp.write('%d\n'%tmp[i,1])
fp.close()
    




