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
import pickle

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

from naca import naca4

# ref:[data,name]
#load airfoil para
path_data='../read_naca_xfoil/outdir/naca_output.txt'
data=np.loadtxt(path_data,skiprows=1,usecols=(1,2,3,4,5,6,7,8))
mypara=data[:,0:3]

#load name
name=[]
with open(path_data, 'r') as infile:
    data=infile.readlines()
    for i in range(1,len(data)):
        name.append(data[i].split()[0])
name=np.asarray(name)
        
#load cl-cl MLP network 
model=load_model('../train_nn/model/final_sf.hdf5')  

#define global variables
global scaler
global tar_cl
global reno
global aoa
global init_cl

#specify fixed variables / values
tar_cl=1.2
scaler=np.array([6,6,30])
reno=np.asarray([50000])/380000.0
aoa=np.asarray([6])/20.0



#specify foil name from which optimization has to start
foil=['naca2408']


for jj in range(len(foil)):
        
    global my_counter
    my_counter =0
    
    global error
    error=[]
    
    def get_coord(p2):
        x,y=naca4(p2*scaler,100)
        return (x,y)
    
    def loss(para):
           
        global my_counter  
        global init_cl
        
        mypara=para
    
        x,y=get_coord(mypara)
    
        my_inp=np.concatenate((reno,aoa,mypara),axis=0)
        my_inp=np.reshape(my_inp,(1,5))
        
        #cd, cl
        out=model.predict([my_inp])
        out=out*np.asarray([0.12,14.0])
                    
        pred_cl=out[0,1]
        
        print ('Pred_cl:', pred_cl)
        if(pred_cl > tar_cl):
            #e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
            e=0
        else:
            e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
        print ('mse:', e)
        
    
        
        if(my_counter == 0):
            init_cl=pred_cl
            fp_conv.write('init-Cl = %s\n'%(init_cl)) 
            fp_conv.write('Iter,   MSE,    Pred_cl\n')
            
        my_counter = my_counter +1
        print ('Iter:', my_counter)
        
        fp_conv.write('%s %s %s\n'%(my_counter,e,pred_cl)) 
           
        plt.figure(figsize=(6,5),dpi=100)
        plt.plot(x,y,'r',label='true')
        plt.ylim([-0.5,0.5])
        plt.savefig('./opti_plot/%s.png'%my_counter,format='png')
        plt.close()
           
        
            
        return  e
    
    #naca2412, 4510,3310,
    fn=foil[jj]     
    path='./result/'
    
    
    fp_conv=open(path+'conv_%s.dat'%fn,'w+')
    
    idx=np.argwhere(name=='%s'%fn)
    
    #naca4510
    p1=mypara[idx[0][0],:]/scaler
    
     
    print('Starting loss = {}'.format(loss(p1)))
    print('Intial foil = %s' %name[idx[0]])
    
    mylimit=((0,1.1),(0,1.1),(0.2,1.1))
    #mylimit=((0,6),(0,6),(6,32))
    res = minimize(loss, x0=p1, method = 'L-BFGS-B', bounds=mylimit, \
                   options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
                                     'eps': 0.01, 'maxfun': 100, \
                                     'maxiter': 50, 'maxls': 100})
    
      
    print('Ending loss = {}'.format(loss(res.x)))
    
    fp=open(path+'final_%s.dat'%fn,'w')
    x,y=get_coord(res.x)
    for i in range(len(x)):
        fp.write('%f %f 0.00\n'%(x[i],y[i]))
    fp.close()
    
    fp=open(path+'base_%s.dat'%fn,'w')
    x00,y00=get_coord(p1)
    for i in range(len(x00)):
        fp.write('%f %f 0.00\n'%(x00[i],y00[i]))
    fp.close()
    
    fp=open(path+'resx_%s.dat'%fn,'w')
    fp.write('Re = %s\n'%(reno*100000))
    fp.write('AoA = %s\n'%(aoa*14))
    fp.write('init-Cl = %s\n'%(init_cl))
    fp.write('tar-Cl = %s\n'%(tar_cl))
    fp.write('%s\n'%res.x)
    fp.write('%s\n'%(res.x*[6,6,30]))
    fp.close()
    
    #intial shape
    x0,y0=get_coord(p1)
    
    
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(x0,y0,'--k',label='Base')
    plt.plot(x,y,'g',lw=3,label='Optimized')
    plt.legend(fontsize=14)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.25,0.25])
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)
    plt.savefig(path+'opti_%s.png'%fn,bbox_inches='tight',dpi=300)
    plt.close()
    
    
    fp_conv.close()  
