#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

optimization using NACA 3 DIgit parameters (naca4_digit_para)

using naca4 flow prediction (case_1_naca_1)

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
from sklearn.metrics import mean_squared_error as skmse
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

from naca import naca4
from get_foil_image import get_foil_mat

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

# ref:[data,name]
#load airfoil para
path='./data_file/'
data_file='naca4_digit_para.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)
para=result1[0]
scaler=result1[1]
name=result1[3]

para=np.asarray(para)
name=np.asarray(name)

#load model para: airfoil coord from parameters
#model_para=load_model('./selected_model/xx.hdf5')  

#load model: predict flow and calcyulate cl
model_flow=load_model('./selected_model/case_1_naca_1/model_sf_150_0.00002427_0.00002472.hdf5') 

#load coordnate xx
global xx
xx=tmp=np.loadtxt('./data_file/xx.txt')

global tar_cl
global reno
global aoa

tar_cl=np.asarray([0.6])
reno=np.asarray([2000])
reno=reno/2000.
aoa=6/14.

#plt.figure(figsize=(6,5),dpi=200)
#plt.plot(reno*200,tar_cl,'-o')
#plt.xlabel('Re',fontsize=20)
#plt.ylabel('Cl',fontsize=20)
##plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#plt.tight_layout()
#plt.savefig('./opt_plot/re_cl.png',format='png')
#plt.close()

pred_cl=np.zeros((len(tar_cl)))
#co=np.zeros((69,2))

global my_counter
my_counter =0


def get_coord(p2):
    x,y=naca4(p2*scaler,100)
    return (x,y)

def loss(p2):
    global my_counter
    #shape (1,16)    
    x,y=get_coord(p2)
    tmp_para=p2
    tmp_para=np.asarray(tmp_para)    
    
    inp_x=x
    inp_y=y
    inp_reno=np.repeat(reno, len(inp_x))
    inp_aoa=np.repeat(aoa, len(inp_x))   
    inp_para=np.repeat(tmp_para[:,None],len(inp_x),axis=1).transpose()   
        
    inp_x=np.asarray(inp_x)
    inp_y=np.asarray(inp_y)
                        
    #reqires shape others-(70,1) & para-(70,16) & val_inp (70,20)    
    val_inp=np.concatenate((inp_x[:,None],inp_y[:,None],inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
    out=model_flow.predict([val_inp]) 
                
    a0=100
    xu=x[:a0+1]
    xl=y[a0:]
       
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
            
    for j in range(len(x)-1):
        xc.append((x[j] + x[j+1])/2.0)
        yc.append((y[j] + y[j+1])/2.0)    
            
        dx.append((x[j+1] - x[j]))
        dy.append((y[j+1] - x[j]))    
        
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
    e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
    print ('mse:', e)
        
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(x,y,'r',label='true')
    plt.ylim([-0.5,0.5])
    plt.savefig('./opt_plot/%s.png'%my_counter,format='png')
    plt.close()
       
    return  e

p1=np.asarray([0.2,0.2,0.5])
p1=para[24]
    
print('Starting loss = {}'.format(loss(p1)))

res = minimize(loss, x0=p1, method = 'L-BFGS-B', \
               options={'disp': True, 'maxcor':20, 'ftol': 1e-16, \
                                 'eps': 0.01, 'maxfun': 20, \
                                 'maxiter': 100, 'maxls': 20})
print('Ending loss = {}'.format(loss(res.x)))

plt.figure(figsize=(6,5),dpi=100)
xi,yi=get_coord(p1)
xo,yo=get_coord(res.x)
plt.plot(xi,yi,'r',label='base')
plt.plot(xo,yo,'g',label='optimized')
plt.ylim([-0.3,0.3])
plt.legend()
plt.savefig('./naca_sp.png',format='png')
plt.close()
