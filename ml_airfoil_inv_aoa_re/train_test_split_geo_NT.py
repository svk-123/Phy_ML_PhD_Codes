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
import random
import os, shutil

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

for ii in [1,2,3,4,5,7,8,9,10]:
#for ii in [9]:         
    # ref:[data,name]
    path='./foil_all_re_aoa/data_files/'
    data_file='data_re_aoa_fp_%s.pkl'%ii
    
    with open(path + data_file, 'rb') as infile:
        result = pickle.load(infile)
    val0=result[0]
    val1=result[1]
    val2=result[2]
    val3=result[3]
    val4=result[4]
    val5=result[5]
    val6=result[6]
    val7=result[7]
    val8=result[8]
    val9=result[9]
    
    
    val0=np.asarray(val0)
    val1=np.asarray(val1)    
    val2=np.asarray(val2)   
    val3=np.asarray(val3)    
    val4=np.asarray(val4)    
    val5=np.asarray(val5)    
    val6=np.asarray(val6)    
    val7=np.asarray(val7)    
    val8=np.asarray(val8)    
    
    
    unique, counts = np.unique(val6, return_counts=True)
    
    np.random.seed(154328)
    
    N= len(unique)
    kk = np.arange(N)
    np.random.shuffle(kk)
    
    #training
    tmp=[]
    for j in kk[:-10]:
        tmp.extend(np.argwhere(val6==unique[j]))
    tmp=np.asarray(tmp)

    I = tmp[:,0].copy()
    
    np.random.shuffle(I)
    n=len(I)
    
    #training
    tr_val0=val0[I][:n]
    tr_val1=val1[I][:n]
    tr_val2=val2[I][:n]
    tr_val3=val3[I][:n]
    tr_val4=val4[I][:n]
    tr_val5=val5
    tr_val6=val6[I][:n]
    tr_val7=val7[I][:n]
    tr_val8=val8[I][:n]
 
    #testing
    tmp=[]
    for j in kk[-10:]:
        tmp.extend(np.argwhere(val6==unique[j]))
    tmp=np.asarray(tmp)
    J = tmp[:,0].copy()
    np.random.shuffle(J)
    n=len(J)    
    
    #testing
    ts_val0=val0[J][:n]
    ts_val1=val1[J][:n]
    ts_val2=val2[J][:n]
    ts_val3=val3[J][:n]
    ts_val4=val4[J][:n]
    ts_val5=val5
    ts_val6=val6[J][:n]
    ts_val7=val7[J][:n]
    ts_val8=val8[J][:n]
        
    outpath='./foil_all_re_aoa/data_files_train_test_NT/'
    
    data1=[tr_val0, tr_val1, tr_val2, tr_val3, tr_val4, tr_val5, tr_val6, tr_val7, tr_val8, val9]
    with open(outpath+'data_re_aoa_fp_NT_tr_%s.pkl'%ii, 'wb') as outfile:
        pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)
    
    data2=[ts_val0, ts_val1, ts_val2, ts_val3, ts_val4, ts_val5, ts_val6, ts_val7, ts_val8, val9]
    with open(outpath+'data_re_aoa_fp_NT_ts_%s.pkl'%ii, 'wb') as outfile:
        pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)
        
    print ('Done %s\n'%ii)





