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

#open pdf file
fp= PdfPages('least_accurate.pdf')

for ii in [1,2,3,4,5,7,8,9,10]:
#for ii in [1]:
    print ii
    #for ii in [1,2,3,4,5,7,8,9,10]:
    data_file='data_re_aoa_fp_NT_E_ts_%d.pkl'%ii
    inp_up=[]
    inp_lr=[]
    my_out=[]
    reno=[]
    aoa=[]
    name=[]
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
      
#    np.random.seed(154328)
#    
#    N= len(xtr1)
#    I = np.arange(N)
#    np.random.shuffle(I)
#    n=30
#    
#    xtr1=xtr1[I][:n]
#    ttr1=ttr1[I][:n]
   
    my_out=ttr1
    
    del inp_up
    del inp_lr
    del result
    
    
    model_test=load_model('./selected_model/model_cnn_500_0.000005_0.000477.hdf5')  
           
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
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx1 = (np.abs(array - value)).argmin()
        return idx1
    
    idx1=find_nearest(train_l2[:,0], 0.4)
    II=II[idx1:]
    
    my_out=my_out[II]
    out=out[II]
    name=name[II]
    reno=reno[II]
    aoa=aoa[II]
        
    
    batch=int( len(out) /6.)
    
    for n in range(batch):
        tmp=6*n
        print tmp

        k=tmp
        fig = plt.figure(figsize=(16, 8))
        
        ax1 = fig.add_subplot(3,2,1)
        ax1.plot(xx,my_out[k][0:35],'ro',label='true')
        ax1.plot(xx,my_out[k][35:],'ro')
        ax1.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
        ax1.plot(xx,out[k][35:],'b',lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.2,0.2])
        plt.text(0.5,-0.18,'%s-Re=%se6-AoA=%s'%(name[k].upper(),reno[k]*3,aoa[k]*12), horizontalalignment='center',fontsize=14)
        #plt.legend(fontsize=16)
        
        k=tmp+1        
        ax2 = fig.add_subplot(3,2,2)
        ax2.plot(xx,my_out[k][0:35],'ro',label='true')
        ax2.plot(xx,my_out[k][35:],'ro')
        ax2.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
        ax2.plot(xx,out[k][35:],'b',lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.2,0.2])
        plt.text(0.5,-0.18,'%s-Re=%se6-AoA=%s'%(name[k].upper(),reno[k]*3,aoa[k]*12), horizontalalignment='center',fontsize=14)
        
        k=tmp+2       
        ax3 = fig.add_subplot(3,2,3)
        ax3.plot(xx,my_out[k][0:35],'ro',label='true')
        ax3.plot(xx,my_out[k][35:],'ro')
        ax3.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
        ax3.plot(xx,out[k][35:],'b',lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.2,0.2])
        plt.text(0.5,-0.18,'%s-Re=%se6-AoA=%s'%(name[k].upper(),reno[k]*3,aoa[k]*12), horizontalalignment='center',fontsize=14)
            
        k=tmp+3        
        ax4 = fig.add_subplot(3,2,4)
        ax4.plot(xx,my_out[k][0:35],'ro',label='true')
        ax4.plot(xx,my_out[k][35:],'ro')
        ax4.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
        ax4.plot(xx,out[k][35:],'b',lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.2,0.2])
        plt.text(0.5,-0.18,'%s-Re=%se6-AoA=%s'%(name[k].upper(),reno[k]*3,aoa[k]*12), horizontalalignment='center',fontsize=14)
        
        k=tmp+4        
        ax5 = fig.add_subplot(3,2,5)
        ax5.plot(xx,my_out[k][0:35],'ro',label='true')
        ax5.plot(xx,my_out[k][35:],'ro')
        ax5.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
        ax5.plot(xx,out[k][35:],'b',lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.2,0.2])       
        plt.text(0.5,-0.18,'%s-Re=%se6-AoA=%s'%(name[k].upper(),reno[k]*3,aoa[k]*12), horizontalalignment='center',fontsize=14)
        
        k=tmp+5       
        ax6 = fig.add_subplot(3,2,6)
        ax6.plot(xx,my_out[k][0:35],'ro',label='true')
        ax6.plot(xx,my_out[k][35:],'ro')
        ax6.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
        ax6.plot(xx,out[k][35:],'b',lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.2,0.2])
        plt.text(0.5,-0.18,'%s-Re=%se6-AoA=%s'%(name[k].upper(),reno[k]*3,aoa[k]*12), horizontalalignment='center',fontsize=14)
                  
        plt.legend(fontsize=12)   
        fp.savefig(fig)
        plt.close()

    
    

    
fp.close()
