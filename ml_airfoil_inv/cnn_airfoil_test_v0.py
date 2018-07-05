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
path='./naca456/'
data_file='data_cp.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp_up=result[0]
inp_lr=result[1]
my_out=result[2]
reno=result[3]
aoa=result[4]
name=result[5]

inp_up=np.asarray(inp_up)
inp_lr=np.asarray(inp_lr)
my_out=np.asarray(my_out)
reno=np.asarray(reno)
aoa=np.asarray(aoa)

N= len(inp_up)
I = np.arange(N)
np.random.shuffle(I)

J=range(100)

inp_up=inp_up[J,:,:]
inp_lr=inp_lr[J,:,:]
my_out=my_out[J,:,:]
reno=reno[J]
aoa=aoa[J]

myname=[]
for nn in range(len(J)):
    myname.append(name[J[nn]])



xtr1=np.concatenate((inp_up[:,:,:,None],inp_lr[:,:,:,None]),axis=3) 
ttr1=np.reshape(my_out,(len(my_out),216,216,1))  

del result
del inp_up
del inp_lr
del my_out

model_test=load_model('./from_nscc/airfoil_inv_cnn_re_aoa/model_enc_cnn_250_0.040_0.051.hdf5')  

        
out=model_test.predict([xtr1])

for k in range(0,100):


    c=out[k,:,:,0].copy()
    for i in range(216):
        for j in range(216):
            if(c[i,j]<=0):
                c[i,j]=0
    c=c/c.max()            

    fig = plt.figure(figsize=(6, 3),dpi=100)
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(ttr1[k,:,:,0],cmap='gray')
    plt.title('true-%s-%se5-%s'%(myname[k],reno[k]*3,aoa[k]*10))
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(out[k,:,:,0],cmap='gray')
    plt.title('pred-%s-%se5-%s'%(myname[k],reno[k]*3,aoa[k]*10))
    
    #ax3 = fig.add_subplot(1,3,3)
    #ax3.imshow(c,cmap='gray')
    #plt.title('filtered')
    
    plt.savefig('./plot_out1/1_val_n%s'%k)
    plt.show()

    '''a=ttr1[k,:,:,0]
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

    list1=[27]
    
    list2=[]
    list2.append(30)
    
    for ii in range(31,50):
        if(ii%3==0):
            list2.append(ii)
            
    for ii in range(51,160):
        if(ii%5==0):
            list2.append(ii)     
            
    for ii in range(161,190):
        if(ii%3==0):
            list2.append(ii)   
            
    list3=[190,191,192,193]
    
    # for true

    tmpf=[]
    f_t=[]
    for p in range(len(list1)):
        tmp1=[]
        for n in range(len(co_t)):
            if(co_t[n,1]==list1[p]):
                tmp1.append(co_t[n,0])
                
        tmpf.append(tmp1)
    tmpf=tmpf[0]    
    f_t=tmpf[np.argmax(a[tmpf,list1[p]])]     
        
       
    tmpr=[]    
    for p in range(len(list3)):
        tmp1=[]        
        for n in range(len(co_t)):

            if(co_t[n,1]==list2[p]):
                tmp1.append(co_t[n,0])
                
        tmpr.append(tmp1)    
        
    tmp=[]
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
            
            for q in range(l-1):
                if (tmp[q]+1 != tmp[q+1]):
                    mkr=q+1
                    
            tmp1=tmp[:mkr]
            tmp2=tmp[mkr:]
            tmp_up.append(tmp1[np.argmax(a[tmp1,list2[p]])])
            tmp_lr.append(tmp2[np.argmax(a[tmp2,list2[p]])])            
            
        elif (len(tmp)%2 ==1):
            l=len(tmp)
            
            for q in range(l-1):
                if (tmp[q]+1 != tmp[q+1]):
                    mkr=q+1
                    
            tmp1=tmp[:mkr]
            tmp2=tmp[mkr:]
            
            tmp_up.append(tmp1[np.argmax(a[tmp1,list2[p]])])
            tmp_lr.append(tmp2[np.argmax(a[tmp2,list2[p]])])         
    t_up=tmp_up
    t_lr=tmp_lr
    
    # for pred
    tmpf=[]
    f_p=[]
    for p in range(len(list1)):
        tmp1=[]
        for n in range(len(co_p)):
            if(co_p[n,1]==list1[p]):
                tmp1.append(co_p[n,0])
                
        tmpf.append(tmp1)
    tmpf=tmpf[0]    
    f_p=tmpf[np.argmax(b[tmpf,list1[p]])]    
   
    tmp=[]
   
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

        if (len(tmp)==2):
            tmp_up.append(tmp[0])
            tmp_lr.append(tmp[1])
            
        elif (len(tmp)%2 ==0):
            l=len(tmp)
            mkr=0
            for q in range(l-1):
                if (tmp[q]+1 != tmp[q+1]):
                    mkr=q+1
                    
            if (mkr!=0):        
                tmp1=tmp[:mkr]
                tmp2=tmp[mkr:]
            else: 
                tmp1=tmp[:l/2]
                tmp2=tmp[l/2:]
            
            tmp_up.append(tmp1[np.argmax(b[tmp1,list2[p]])])
            tmp_lr.append(tmp2[np.argmax(b[tmp2,list2[p]])])            
            
        elif (len(tmp)%2 ==1):
            l=len(tmp)
            mkr=0
            for q in range(l-1):
                if (tmp[q]+1 != tmp[q+1]):
                    mkr=q+1
                    
            if (mkr!=0):        
                tmp1=tmp[:mkr]
                tmp2=tmp[mkr:]
            else: 
                tmp1=tmp[:l/2]
                tmp2=tmp[l/2:]

            tmp_up.append(tmp1[np.argmax(b[tmp1,list2[p]])])
            tmp_lr.append(tmp2[np.argmax(b[tmp2,list2[p]])])       
            
    p_up=tmp_up
    p_lr=tmp_lr
    
    pix=1/167.0
    list1=np.asarray(list1)*pix
    f_t=np.asarray(f_t)*pix
    f_p=np.asarray(f_p)*pix
    
    list2=np.asarray(list2)*pix
    t_up=np.asarray(t_up)*pix
    t_lr=np.asarray(t_lr)*pix
    p_up=np.asarray(p_up)*pix
    p_lr=np.asarray(p_lr)*pix    
    
    newlist=np.zeros((len(list1)+len(list2)))
    newlist[:len(list1)]=list1
    newlist[len(list1):]=list2    

    nt_up=np.zeros((len(list1)+len(list2)))
    nt_up[:len(list1)]=f_t
    nt_up[len(list1):]=t_up      

    nt_lr=np.zeros((len(list1)+len(list2)))
    nt_lr[:len(list1)]=f_t
    nt_lr[len(list1):]=t_lr      
 
    np_up=np.zeros((len(list1)+len(list2)))
    np_up[:len(list1)]=f_p
    np_up[len(list1):]=p_up      

    np_lr=np.zeros((len(list1)+len(list2)))
    np_lr[:len(list1)]=f_p
    np_lr[len(list1):]=p_lr   
    
    fig = plt.figure(figsize=(9, 3),dpi=100)
    
    newlist=newlist-27*pix
    nt_up=nt_up-f_t
    nt_lr=nt_lr-f_t
    np_up=np_up-f_t
    np_lr=np_lr-f_t    
    
    
    
    plt.plot(newlist,nt_up,'-ro',lw=2,label='true')
    plt.plot(newlist,nt_lr,'-ro',lw=2)
    plt.plot(newlist,np_up,'b',lw=2,label='prediction')
    plt.plot(newlist,np_lr,'b',lw=2)    
    plt.legend()
    plt.savefig('%s.png'%k)
    plt.show()'''








