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
from matplotlib.backends.backend_pdf import PdfPages
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
# ref:[data,name]
path='./foil_all_re_aoa/data_files_train_test_NT/'



for ii in [5]:
#for ii in [1,2,3,4,5,7,8,9,10]:
    data_file='data_re_aoa_fp_NT_tr_%d.pkl'%ii
    inp_up=[]
    inp_lr=[]
    out=[]
    reno=[]
    aoa=[]
    name=[]
    with open(path + data_file, 'rb') as infile:
        result = pickle.load(infile)
    print result[-1:]    
    
    inp_up.extend(result[0])
    inp_lr.extend(result[1])
    out.extend(result[2])
    reno.extend(result[3])
    aoa.extend(result[4])
    name.extend(result[6])
    
    name=np.asarray(name)
    
    inp_up=np.asarray(inp_up)
    inp_lr=np.asarray(inp_lr)
    out=np.asarray(out)
    xx=result[5]
    
    xtr1=np.concatenate((inp_up[:,:,:,None],inp_lr[:,:,:,None]),axis=3) 
    ttr1=out 
    
    my_out=out.copy()
    
    del inp_up
    del inp_lr
    del out
    del result
    
    
    model_test=load_model('./selected_model/case_2/model_cnn/final_cnn.hdf5')  
           
    out=model_test.predict([xtr1])
    out=out*0.18
    
    np.random.seed(12535)
    I = np.arange(out.shape[0])
    np.random.shuffle(I)
    n=out.shape[0]
    out=out[I][:n]
    my_out=my_out[I][:n]
    name=name[I][:n]
    
#    for k in range(1,109,10):
#        print k
#        plt.figure(figsize=(6,5),dpi=100)
#        plt.plot(xx,my_out[k][0:35],'ro',markersize=8,label='true')
#        plt.plot(xx,my_out[k][35:],'ro',markersize=8)
#        plt.plot(xx,out[k][0:35],'b',lw=3,label='CNN')
#        plt.plot(xx,out[k][35:],'b',lw=3)
#        plt.xlim([-0.05,1.05])
#        plt.ylim([-0.2,0.2])
#        plt.legend(fontsize=20)
#        plt.xlabel('X',fontsize=20)
#        plt.ylabel('Y',fontsize=20)  
#        #plt.axis('off')
#        plt.tight_layout()
#        plt.savefig('./plot2/tr_%s_%s.png'%(k,name[k]), bbox_inches='tight',dpi=100)
#        plt.close()
        
        
    tmp=0
    mm=0    
    for mm in range(100):
        k=tmp+mm
        print k
        fig = plt.figure(figsize=(6, 5))
        plt.plot(xx,my_out[k][0:35],'o',mfc='grey',mec='grey',ms=10,label='true')
        plt.plot(xx,my_out[k][35:],'o',mfc='grey',mec='grey',ms=10)
        plt.plot(xx,out[k][0:35],'k',lw=3,label='prediction')
        plt.plot(xx,out[k][35:],'k',lw=3)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.2,0.2])
        
        plt.subplots_adjust(top = 0.9, bottom = 0.1, right = 0.9, left = 0.1, hspace = 0.3, wspace = 0.2)
        plt.text(0.5,-0.18,'%s, Re=%se6, AoA=%s %%'%(name[k].upper(),reno[k]*3,aoa[k]*12), horizontalalignment='center',fontsize=14)            
        plt.xlabel('X',fontsize=24)
        plt.ylabel('Y',fontsize=24)     
        plt.legend(fontsize=14,frameon=False)        
        plt.savefig('./plot2/ts_%s_%s.png'%(k,name[k]), bbox_inches='tight',dpi=100)
        plt.close()
        
        