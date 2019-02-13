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
# ref:[data,name]
path='./'
data_file='train_output_with_l2.pkl'


with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
print result[-1:]

my_out=result[0]
out=result[1]
reno=result[2]
aoa=result[3]
xx=result[4]
name=result[5]
train_l2=result[6]

##error plot
#plt.figure(figsize=(6,5),dpi=200)
#plt.hist(train_l2[:,0], 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
#plt.xlabel('$L_2$ relative error(%)',fontsize=20)
#plt.ylabel('Number of Samples',fontsize=20)
#plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#plt.savefig('train_error.png', bbox_inches='tight',dpi=200)
#plt.show()
#
#print ('avg l2:',sum(train_l2[:,0])/len(train_l2[:,0]))



#open pdf file
fp= PdfPages('train_result_1.pdf')

batch=int( len(out) /18.)


for jj in range(1):    
    for n in range(1620,1631):
        
        tmp=18*n
        print tmp
            
        
        fig = plt.figure(figsize=(24, 24))
        for mm in range(18):
            k=tmp+mm
            print k
            ax1 = fig.add_subplot(6,3,mm+1)
            ax1.plot(xx,my_out[k][0:35],'o',mfc='grey',mec='grey',ms=10,label='true')
            ax1.plot(xx,my_out[k][35:],'o',mfc='grey',mec='grey',ms=10)
            ax1.plot(xx,out[k][0:35],'k',lw=3,label='prediction')
            ax1.plot(xx,out[k][35:],'k',lw=3)
            plt.xlim([-0.05,1.05])
            plt.ylim([-0.2,0.2])
            plt.subplots_adjust(top = 0.9, bottom = 0.1, right = 0.9, left = 0.1, hspace = 0.3, wspace = 0.2)
            plt.text(0.5,-0.18,'%s, Re=%se6, AoA=%s, E=%0.4f %%'%(name[k].upper(),reno[k]*3,aoa[k]*12,train_l2[k,0]), horizontalalignment='center',fontsize=14)            
            
        plt.legend(fontsize=14,frameon=False)   
        fp.savefig(fig)
        plt.close()
            
      
#    tmp=18*batch
#    print tmp
#
#    fig = plt.figure(figsize=(24, 24))
#    for mm in range(5):
#        k=tmp+mm
#        print k
#        ax1 = fig.add_subplot(6,3,mm+1)
#        ax1.plot(xx,my_out[k][0:35],'o',mfc='grey',mec='grey',ms=10,label='true')
#        ax1.plot(xx,my_out[k][35:],'o',mfc='grey',mec='grey',ms=10)
#        ax1.plot(xx,out[k][0:35],'k',lw=3,label='prediction')
#        ax1.plot(xx,out[k][35:],'k',lw=3)
#        plt.xlim([-0.05,1.05])
#        plt.ylim([-0.2,0.2])
#        plt.subplots_adjust(top = 0.9, bottom = 0.1, right = 0.9, left = 0.1, hspace = 0.3, wspace = 0.2)
#        plt.text(0.5,-0.18,'%s, Re=%se6, AoA=%s, E=%0.4f %%'%(name[k].upper(),reno[k]*3,aoa[k]*12,train_l2[k,0]), horizontalalignment='center',fontsize=14)
    
    
    plt.legend(fontsize=14,frameon=False)   
    fp.savefig(fig)
    plt.close()
  
    

    
fp.close()
