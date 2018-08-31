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
path='./foil_all_re_aoa/data_files/'
data_file='data_re_aoa_fp_4.pkl'



for ii in [9]:
#for ii in [1,2,3,4,5,7,8,9,10]:
    data_file='data_re_aoa_fp_%d.pkl'%ii
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
    
    
    model_test=load_model('./selected_model/for_choosing_foil/model_enc_cnn_200_0.000321_0.000794.hdf5')  
           
    out=model_test.predict([xtr1])
    out=out*0.18
    

    for k in range(10):
        print k
    #    plt.figure(figsize=(6,5),dpi=100)
    #    plt.plot(xx,my_out[k][0:35],'ro',markersize=8,label='true')
    #    plt.plot(xx,my_out[k][35:],'ro',markersize=8)
    #    plt.plot(xx,out[k][0:35],'b',lw=3,label='CNN')
    #    plt.plot(xx,out[k][35:],'b',lw=3)
    #    plt.xlim([-0.05,1.05])
    #    plt.ylim([-0.2,0.2])
    #    plt.legend(fontsize=20)
    #    plt.xlabel('X',fontsize=20)
    #    plt.ylabel('Y',fontsize=20)  
    #    #plt.axis('off')
    #    plt.tight_layout()
    #    plt.savefig('./plot/ts_%s_%s.png'%(k,name[k]), bbox_inches='tight',dpi=100)
    #    plt.close()
        
        
        fig = plt.figure(figsize=(8, 4),dpi=100)
        
        ax1 = fig.add_subplot(6,1,1)
        ax1.plot(xx,my_out[k][0:35],'ro',label='true')
        ax1.plot(xx,my_out[k][35:],'ro')
        ax1.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
        ax1.plot(xx,out[k][35:],'b',lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.2,0.2])
        plt.legend(fontsize=16)
     
        ax2 = fig.add_subplot(6,1,2)
        ax2.plot(xx,my_out[k][0:35],'ro',label='true')
        ax2.plot(xx,my_out[k][35:],'ro')
        ax2.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
        ax2.plot(xx,out[k][35:],'b',lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-2.5,1])
        plt.legend(fontsize=16)
        
        
        ax3 = fig.add_subplot(6,2,1)
        ax3.plot(xx,my_out[k][0:35],'ro',label='true')
        ax3.plot(xx,my_out[k][35:],'ro')
        ax3.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
        ax3.plot(xx,out[k][35:],'b',lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-2.5,1])
        plt.legend(fontsize=16)        
        
        
        ax4 = fig.add_subplot(6,2,2)
        ax4.plot(xx,my_out[k][0:35],'ro',label='true')
        ax4.plot(xx,my_out[k][35:],'ro')
        ax4.plot(xx,out[k][0:35],'b',lw=2,label='prediction')
        ax4.plot(xx,out[k][35:],'b',lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-2.5,1])
        plt.legend(fontsize=16)        
        
        
        
        plt.savefig()
        plt.show()
    
    
    
    
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
    #plt.plot(my_out[0],out[0],'ro')
    #for k in range(len(name)):
    #    
    #    plt.plot(my_out[k],out[k],'ro')
    #plt.legend(fontsize=20)
    #plt.xlabel('True',fontsize=20)
    #plt.ylabel('Prediction',fontsize=20)
    #plt.xlim([-0.20,0.20])
    #plt.ylim([-0.20,0.20])    
    #plt.savefig('train_spread.png', bbox_inches='tight',dpi=100)
    #plt.show()          
    
    #error plot
    plt.figure(figsize=(6,5),dpi=100)
    plt.hist(train_l2, 20,histtype='bar', stacked=True)
    plt.xlabel('L2 relative error(%)',fontsize=20)
    plt.ylabel('number of Samples',fontsize=20)
    plt.savefig('train_error_%s.png'%ii, bbox_inches='tight',dpi=100)
    plt.show()
    
    #get foil name with lease error
    num=np.asarray(range(len(train_l2)))
    train_l2=np.asarray(train_l2)
    tmp=np.concatenate((train_l2[:,None],num[:,None]),axis=1)
    tmp = tmp[tmp[:,0].argsort()]
        
    
    fp=open('best_of_foils_%s.dat'%ii,'w+')
    for i in range(len(name)):
        fp.write('%s\n'%name[int(tmp[i,1])])
    fp.close()
    
    
    
    
    
    
    
    '''path='./selected_model/case_3_fp'
    data_file='/hist.pkl'
    with open(path + data_file, 'rb') as infile:
        result = pickle.load(infile)
    history=result[0]
    #hist
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(range(len(history['loss'])),history['loss'],'r',lw=3,label='training_error')
    plt.plot(range(len(history['val_loss'])),history['val_loss'],'b',lw=3,label='validation_error')
    plt.legend(fontsize=20)
    plt.xlabel('Training Epochs',fontsize=20)
    plt.ylabel('MSE',fontsize=20)
    plt.yscale('log')
    #plt.xlim([-0.05,1.05])
    #plt.ylim([-0.2,0.2])    
    plt.savefig('convergence.png', bbox_inches='tight',dpi=100)
    plt.show()'''
    
    

