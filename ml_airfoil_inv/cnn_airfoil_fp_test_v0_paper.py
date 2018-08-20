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
from scipy.interpolate import interp1d
 
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
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

data_file='data_144_1600_ts.pkl'

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
name=np.asarray(name)

xtr1=np.concatenate((inp_up[:,:,:,None],inp_lr[:,:,:,None]),axis=3) 
ttr1=my_out 

tmp=np.loadtxt('ts_30.dat')
tmp=tmp.astype(int)

xtr1=xtr1[tmp]
ttr1=ttr1[tmp]
my_out=my_out[tmp]
name=name[tmp]

#model
'''model_test=load_model('./hyper_selected/case144/case_1/final_enc_cnn.hdf5')  
out1=model_test.predict([xtr1])
out1=out1*0.18

model_test=load_model('./hyper_selected/case144/case_2/final_enc_cnn.hdf5')  
out2=model_test.predict([xtr1])
out2=out2*0.18'''

model_test=load_model('./hyper_selected/case144/case_3/final_enc_cnn.hdf5')   
out=model_test.predict([xtr1])
out=out*0.18

#plot one CNN
for k in range(len(name)):

    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx,my_out[k][0:35],'o',mfc='grey',mec='grey',ms=10,label='true')
    plt.plot(xx,my_out[k][35:],'o',mfc='grey',mec='grey',ms=10,)
    plt.plot(xx,out[k][0:35],'k',lw=3,label='CNN')
    plt.plot(xx,out[k][35:],'k',lw=3)
#    
#    f1 = interp1d(xx,out[k][0:35], kind='cubic')
#    f2 = interp1d(xx,out[k][35:], kind='cubic')
#
#    plt.plot(xx,f1(xx),'r',lw=2,label='CNN_sp')
#    plt.plot(xx,f2(xx),'r',lw=2)    
    
      

    plt.xlim([-0.05,1.05])
    plt.ylim([-0.2,0.2])
    plt.legend(loc="upper left", bbox_to_anchor=[0.0, 1], ncol=2, fontsize=24, \
               frameon=False, shadow=False, fancybox=False,title='')
    plt.xlabel('X',fontsize=24)
    plt.ylabel('Y',fontsize=24)  
    #plt.axis('off')
    plt.tight_layout()
    plt.savefig('./plot/ts_%s_%s.png'%(k,name[k]), format='png', bbox_inches='tight',dpi=100)
    plt.show()
    

'''#plot multiple
for k in range(10):

    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx,my_out[k][0:35],'o',mfc='grey',mec='grey',ms=10,label='true')
    plt.plot(xx,my_out[k][35:],'o',mfc='grey',mec='grey',ms=8,)
    
    #plt.plot(xx,out1[k][0:35],'k',lw=2,label='CNN-1')
    #plt.plot(xx,out1[k][35:],'k',lw=2)
    
    #plt.plot(xx,out2[k][0:35],'b',lw=2,label='CNN-2')
    #plt.plot(xx,out2[k][35:],'b',lw=2)
    
    plt.plot(xx,out3[k][0:35],'k',lw=3,label='CNN-3')
    plt.plot(xx,out3[k][35:],'k',lw=3)
    
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.2,0.2])
    plt.legend(loc="upper left", bbox_to_anchor=[0.1, 1], ncol=2, fontsize=16, \
               frameon=False, shadow=False, fancybox=False,title='')
    plt.xlabel('X',fontsize=20)
    plt.ylabel('Y',fontsize=20)  
    #plt.axis('off')
    plt.tight_layout()
    plt.savefig('./plot/tr_%s_%s.eps'%(k,name[k]), format='eps', bbox_inches='tight',dpi=100)
    plt.show()'''




