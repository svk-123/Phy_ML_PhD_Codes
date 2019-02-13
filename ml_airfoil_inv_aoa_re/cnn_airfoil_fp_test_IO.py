#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017
write all the predictions to file with error
then plot other way
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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

# ref:[data,name]
# ref:[data,name]
path='./foil_all_re_aoa/data_files_train_test_NT/'
data_file='data_re_aoa_fp_4.pkl'
inp_up=[]
inp_lr=[]
my_out=[]
reno=[]
aoa=[]
name=[]

for ii in [1,2,3,4,5,6,7,8,9,10]:
#for ii in [1]:
    print ii
    data_file='data_re_aoa_fp_NT_E_ts_%d.pkl'%ii

    with open(path + data_file, 'rb') as infile:
        result = pickle.load(infile)
    print result[-1:]    
    
    inp_up.extend(result[0])
    inp_lr.extend(result[1])
    my_out.extend(result[2])
    reno.extend(result[3])
    aoa.extend(result[4])
    name.extend(result[6])
    
inp_up=np.asarray(inp_up)
inp_lr=np.asarray(inp_lr)
my_out=np.asarray(my_out)
name=np.asarray(name)
reno=np.asarray(reno)
aoa=np.asarray(aoa)
    
xx=result[5]
    
xtr1=np.concatenate((inp_up[:,:,:,None],inp_lr[:,:,:,None]),axis=3) 
ttr1=my_out 
  
  
del inp_up
del inp_lr
del result
    
    
model_test=load_model('./selected_model/case_1/model_cnn_575_0.000003_0.000457.hdf5')  
           
out=model_test.predict([xtr1])
out=out*0.18

#calculate error norm
train_l2=[]
train_l1=[]
for k in range(len(out)):    
        
    tmp=my_out[k]-out[k]
    train_l2.append( (LA.norm(tmp)/LA.norm(out))*100 )
    tmp2=tmp/out[k]
    train_l1.append(sum(abs(tmp2))/len(out))    


train_l2=np.asarray(train_l2)
idx=range(0,len(out))
idx=np.asarray(idx)
train_l2=np.concatenate((train_l2[:,None],idx[:,None]),axis=1)
train_l2=train_l2[train_l2[:,0].argsort()]
II=train_l2[:,1].astype(int)
    
out=out[II]
my_out=my_out[II]    
name=name[II]
reno=reno[II]
aoa=aoa[II]    

outpath='./'
info='my_out,out,reno,aoa,xx,name,train_l2,info'    
data1=[my_out,out,reno,aoa,xx,name,train_l2,info]
with open(outpath+'test_output_with_l2.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)        

