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

train_error=[]
#spread_plot
plt.figure(figsize=(6,5),dpi=100)

#1
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

model_test=load_model('./hyper_selected/case144/case_1/final_enc_cnn.hdf5')  
       
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

train_l2=np.sort(train_l2)

plt.plot(range(len(train_l2)),train_l2,'g',label='144-CNN-1')

#2
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

model_test=load_model('./hyper_selected/case144/case_2/final_enc_cnn.hdf5')  
       
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

train_l2=np.sort(train_l2)

plt.plot(range(len(train_l2)),train_l2,'b',label='144-CNN-2')

#3
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

train_l2=np.sort(train_l2)

plt.plot(range(len(train_l2)),train_l2,'r',label='144-CNN-3')



#4
data_file='data_216_1600_tr.pkl'
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

model_test=load_model('./hyper_selected/case216/case_1/final_enc_cnn.hdf5')  
       
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

train_l2=np.sort(train_l2)

plt.plot(range(len(train_l2)),train_l2,'k',label='216-CNN-1')


#5
data_file='data_216_1600_tr.pkl'
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

model_test=load_model('./hyper_selected/case216/case_2/final_enc_cnn.hdf5')  
       
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

train_l2=np.sort(train_l2)

plt.plot(range(len(train_l2)),train_l2,'orange',label='216-CNN-2')


#6
data_file='data_216_1600_tr.pkl'
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

model_test=load_model('./hyper_selected/case216/case_3/final_enc_cnn.hdf5')  
       
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

train_l2=np.sort(train_l2)

plt.plot(range(len(train_l2)),train_l2,'c',label='216-CNN-3')










    
plt.legend(fontsize=12)
plt.xlabel('Sample-no',fontsize=20)
plt.ylabel('Error',fontsize=20)
plt.ylim([-10,500])   
plt.ylim([-0.05,1.0])    
plt.savefig('tr_spread.eps', format='eps', bbox_inches='tight',dpi=100)
plt.show()          

'''#get foil name with lease error
num=np.asarray(range(len(train_l2)))
train_l2=np.asarray(train_l2)
tmp=np.concatenate((train_l2[:,None],num[:,None]),axis=1)
tmp = tmp[tmp[:,0].argsort()]
fp=open('ts_foil_no_144.dat','w+')
for i in range(10):
    fp.write('%d\n'%tmp[i,1])
fp.close() '''
    




