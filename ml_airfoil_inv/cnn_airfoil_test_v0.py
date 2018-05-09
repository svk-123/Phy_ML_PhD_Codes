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
path='./'
data_file='data_cp.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp_up=result[0]
inp_lr=result[1]
my_out=result[2]
name=result[3]

inp_up=np.asarray(inp_up)
inp_lr=np.asarray(inp_lr)
my_out=np.asarray(my_out)

'''inp_up=inp_up[0:100,:,:]
inp_lr=inp_lr[0:100,:,:]
my_out=my_out[0:100,:,:]'''

xtr1=np.concatenate((inp_up[:,:,:,None],inp_lr[:,:,:,None]),axis=3) 
ttr1=np.reshape(my_out,(len(my_out),216,216,1))  

'''del result
del inp_up
del inp_lr
del my_out'''

model_test=load_model('./from_nscc/airfoil_inv_cnn_sig_bce/model_cnn_sig_bce/model_enc_cnn_1000_0.024_0.065.hdf5')  

        
out=model_test.predict([xtr1])

for k in range(0,1):


    c=out[k,:,:,0].copy()
    for i in range(216):
        for j in range(216):
            if(c[i,j]<=0):
                c[i,j]=0
    c=c/c.max()            

    '''fig = plt.figure(figsize=(6*5, 3*5),dpi=100)
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(ttr1[k,:,:,0],cmap='gray')
    plt.title('true-%s'%name[k])
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(out[k,:,:,0],cmap='gray')
    plt.title('pred-%s'%name[k])
    
    #ax3 = fig.add_subplot(1,3,3)
    #ax3.imshow(c,cmap='gray')
    #plt.title('filtered')
    
    plt.savefig('./plot_out/val_n%s'%k)
    plt.show()'''

    a=ttr1[k,:,:,0]
    b=out[k,:,:,0]
    # for column
    for n in range(216):
        if (np.all(a[:,n]==0) != True):
            c1=n
            break
        
    for n in range(216):    
        if (np.all(a[:,216-n-1]==0) != True):
            c2=216-n-1
            break        
            
    # for row
    for n in range(216):
        if (np.all(a[n,:]==0) != True):
            r1=n
            break
        
    for n in range(216):    
        if (np.all(a[216-n-1,:]==0) != True):
            r2=216-n-1
            break             
    #tmp=np.argwhere(a > 0.9)
    tmp=np.nonzero(a > 0.6)
    co_t=np.concatenate((tmp[0][:,None],tmp[1][:,None]),axis=1)
    co_t=co_t[np.argsort(co_t[:,1])]
    
    tmp=np.nonzero(b > 0.2)
    co_p=np.concatenate((tmp[0][:,None],tmp[1][:,None]),axis=1)
    co_p=co_p[np.argsort(co_p[:,1])]

    list1=[29]
    list2=[35,40,45,50,55,60,80,100,120,140,160,180,185]
    list3=[190]
    
    # for true
    tmp=[]
    '''for p in range(len(list1)):
        tmp1=[]
        for n in range(len(co_t)):
            if(co_t[n,1]==list1[p]):
                tmp1.append(co_t[n,0])
                
        tmp.append(tmp1)      '''  
        
    '''for p in range(len(list3)):
        tmp1=[]        
        for n in range(len(co_t)):

            if(co_t[n,1]==list2[p]):
                tmp1.append(co_t[n,0])
                
        tmp.append(tmp1)'''    
    
    for p in range(len(list2)):
        tmp1=[]
        for n in range(len(co_t)):

            if(co_t[n,1]==list2[p]):
                tmp1.append(co_t[n,0])
        tmp.append(tmp1)        
    val=tmp    
    
    tmp_up=[]
    tmp_lr=[]    
    for p in range(len(val)):
        tmp=val[p]
        
        tmp=np.sort(tmp)
        
       
        if (len(tmp)==2):
            tmp_up.append(tmp[0])
            tmp_lr.append(tmp[1])
            
        elif (len(tmp)%2 ==0):
            l=len(tmp)
            tmp1=tmp[:l/2]
            tmp2=tmp[-l/2:]
            tmp_up.append(tmp1[np.argmax(a[tmp1,list2[p]])])
            tmp_lr.append(tmp2[np.argmax(a[tmp2,list2[p]])])            
            
        elif (len(tmp)%2 ==1):
            l=len(tmp)
            
            for q in range(l-1):
                if (tmp[q]+1 != tmp[q+1]):
                    mkr=q
            tmp1=tmp[:mkr]
            tmp2=tmp[mkr:]
            tmp_up.append(tmp1[np.argmax(a[tmp1,list2[p]])])
            tmp_lr.append(tmp2[np.argmax(a[tmp2,list2[p]])])         
    t_up=tmp_up
    t_lr=tmp_lr
    
    # for pred
    tmp=[]
    '''for p in range(len(list1)):
        tmp1=[]
        for n in range(len(co_t)):
            if(co_t[n,1]==list1[p]):
                tmp1.append(co_t[n,0])
                
        tmp.append(tmp1)      '''  
        
    '''for p in range(len(list3)):
        tmp1=[]        
        for n in range(len(co_t)):

            if(co_t[n,1]==list2[p]):
                tmp1.append(co_t[n,0])
                
        tmp.append(tmp1)'''    
    
    for p in range(len(list2)):
        tmp1=[]
        for n in range(len(co_p)):

            if(co_p[n,1]==list2[p]):
                tmp1.append(co_p[n,0])
        tmp.append(tmp1)        
    val=tmp    
    
    tmp_up=[]
    tmp_lr=[]    
    for p in range(len(val)):
        tmp=val[p]
        
        tmp=np.sort(tmp)
        print tmp
        if (len(tmp)==2):
            tmp_up.append(tmp[0])
            tmp_lr.append(tmp[1])
            
        elif (len(tmp)%2 ==0):
            l=len(tmp)
            for q in range(l-1):
                if (tmp[q]+1 != tmp[q+1]):
                    mkr=q
            tmp1=tmp[:mkr]
            tmp2=tmp[mkr:]
            
            tmp_up.append(tmp1[np.argmax(b[tmp1,list2[p]])])
            tmp_lr.append(tmp2[np.argmax(b[tmp2,list2[p]])])            
            
        elif (len(tmp)%2 ==1):
            l=len(tmp)
            
            for q in range(l-1):
                if (tmp[q]+1 != tmp[q+1]):
                    mkr=q
            tmp1=tmp[:mkr]
            tmp2=tmp[mkr:]
            tmp_up.append(tmp1[np.argmax(b[tmp1,list2[p]])])
            tmp_lr.append(tmp2[np.argmax(b[tmp2,list2[p]])])       
            
    p_up=tmp_up
    p_lr=tmp_lr
    
    pix=1/167.0
    list2=np.asarray(list2)*pix
    t_up=np.asarray(t_up)*pix
    t_lr=np.asarray(t_lr)*pix
    p_up=np.asarray(p_up)*pix
    p_lr=np.asarray(p_lr)*pix    
    
    
    fig = plt.figure(figsize=(9, 3),dpi=100)
    plt.plot(list2,t_up,'o')
    plt.plot(list2,t_lr,'o')
    plt.plot(list2,p_up,'-o')
    plt.plot(list2,p_lr,'-o')    
    plt.show()








