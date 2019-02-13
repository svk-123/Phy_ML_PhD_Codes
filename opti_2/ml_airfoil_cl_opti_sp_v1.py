#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

with new parameters tanh_16_v1:

with new flow prediction network using v1.

@author: vinoth
"""
#based on parameters 

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
from keras.layers import merge, Input, dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import cPickle as pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

from numpy import linalg as LA
import os, shutil
from skimage import io, viewer,util 
from scipy.optimize import minimize

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 


# ref:[data,name]
#load airfoil para
path='./data_file/'
data_file='param_216_16.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)
para=result1[0][0]
name=result1[1]
para=np.asarray(para)
name=np.asarray(name)

#load model para: airfoil coord from parameters
model_para=load_model('./selected_model/p16/model_cnn_2950_0.000013_0.000176.hdf5')  

#load model: predict flow and calcyulate cl
model_flow=load_model('./selected_model/case_9_naca_lam_1/model_sf_65_0.00000317_0.00000323.hdf5') 

#load coordnate xx
global xx
xx=tmp=np.loadtxt('./data_file/xx.txt')

global tar_cl
global reno
global aoa

tar_cl=0.18
reno=600/2000.
aoa=6/14.

global my_counter
my_counter =0

def loss(para):
    global my_counter    
    my_para=np.reshape(para,(1,16))
    tmp_para=para
    #get coord from para 
    # requires input shape (1,16)
    get_coord= K.function([model_para.layers[12].input],[model_para.layers[15].output])
    c1 = get_coord([my_para])[0][0,:]
    co=np.zeros((69,2))
    co[0:35,0]=xx[::-1]
    co[0:35,1]=c1[:35][::-1]
    co[35:69,0]=xx[1:]
    co[35:69,1]=c1[36:]
    
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(co[:,0],co[:,1],'r',label='true')
    plt.savefig('./opt_plot/%s.png'%my_counter,format='png')
    plt.close()
    
    #get flow field 
    inp_x=co[:,0]
    inp_y=co[:,1]
   
    inp_reno=np.repeat(reno, len(inp_x))
    inp_aoa=np.repeat(aoa, len(inp_x))   
    inp_para=np.repeat(tmp_para[:,None],len(inp_x),axis=1).transpose()   
    #reqires shape others-(70,1) & para-(70,16) & val_inp (70,20)    
    val_inp=np.concatenate((inp_x[:,None],inp_y[:,None],inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
    out=model_flow.predict([val_inp]) 
    
            
    #a0=find_nearest(co[:,0],0)
    a0=34
    xu=co[:a0+1,0]
    yu=co[:a0+1,1]
    xl=co[a0:,0]
    yl=co[a0:,1]
        
    #pressure
    pu2=out[:a0+1,0]
    pl2=out[a0:,0]
                  
    #cl calculation        
    xc=[]
    yc=[]
    dx=[]
    dy=[]
        
    pc=[]
    for j in range(len(xu)-1): 
        pc.append((pu2[j]+pu2[j+1])/2.0)
    for j in range(len(xl)-1): 
        pc.append((pl2[j]+pl2[j+1])/2.0)
            
    for j in range(len(co)-1):
        xc.append((co[j,0] + co[j+1,0])/2.0)
        yc.append((co[j,1] + co[j+1,1])/2.0)    
            
        dx.append((co[j+1,0] - co[j,0]))
        dy.append((co[j+1,1] - co[j,1]))    
        
    cp=[]    
    for j in range(len(xc)):
        cp.append(2*pc[j])
         
    lF=[]
    for j in range(len(xc)):
        if(dx[j] <=0):
            lF.append(-0.5*cp[j]*abs(dx[j]))
        else:                
            lF.append(0.5*cp[j]*abs(dx[j]))
                
    pred_cl=sum(lF)*np.cos(np.radians(aoa))/(0.5)
    
    print ('Pred_cl:', pred_cl)
    
    
    my_counter = my_counter +1
    print ('Iter:', my_counter)
    
    return  abs(tar_cl-pred_cl)
     

idx=np.argwhere(name=='naca0024')[0][0]
p1=para[idx,:]
    
print('Starting loss = {}'.format(loss(p1)))

res = minimize(loss, x0=p1, method = 'L-BFGS-B', \
               options={'disp': True, 'maxcor':10, 'ftol': 1e-16, \
                                 'eps': 1e-2, 'maxfun': 10, \
                                 'maxiter': 20, 'maxls': 10})
print('Ending loss = {}'.format(loss(res.x)))


my_para=np.reshape(res.x,(1,16))
# requires input shape (1,16)
get_coord= K.function([model_para.layers[12].input],[model_para.layers[15].output])
c1 = get_coord([my_para])[0][0,:]
co=np.zeros((69,2))
co[0:35,0]=xx[::-1]
co[0:35,1]=c1[:35][::-1]
co[35:69,0]=xx[1:]
co[35:69,1]=c1[36:]
plt.figure(figsize=(6,5),dpi=100)
plt.plot(co[:,0],co[:,1],'r',label='true')
plt.show()
