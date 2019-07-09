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
path='./data_file/'
data_file='naca4_clcd_turb_st_3para.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile, encoding='bytes')

mycd=result1[1]
mycl=result1[2]
myreno=result1[3]
myaoa=result1[4]
mypara=result1[5]
name=result1[6]

mypara=np.asarray(mypara)
name=np.asarray(name)

nname=[]
for i in range(len(name)):
    nname.append(name[i].decode())
name=np.asarray(nname)

#fp=open('cl_data_turb_st.txt','w')
#for i in range(len(mycl)):
#    fp.write('%s %f %f %f %f %f \n'%(name[i],myreno[i],myaoa[i],mycl[i],mycd[i],mycl[i]/mycd[i]))
#    
#fp.close()
    

#load model para: airfoil coord from parameters
#scaler used in cl pred network
# reno=10000
#aoa=14
#para=np.array([6,6,30])
#cl=0.8
#cd=0.25

model=load_model('./selected_model/turb_3para_st_6x30/final_sf.hdf5')  
global scaler
scaler=np.array([6,6,30])

global tar_clcd
global reno
global aoa

tar_clcd=26
reno=np.asarray([50000])/100000.
aoa=np.asarray([6])/14.

reno=np.asarray(reno)
aoa=np.asarray(aoa)

global my_counter
my_counter =0

global error
error=[]

def get_coord(p2):
    x,y=naca4(p2*scaler,100)
    return (x,y)

def loss(para):
       
    global my_counter  
    mypara=para

    x,y=get_coord(mypara)

    my_inp=np.concatenate((reno,aoa,mypara),axis=0)
    my_inp=np.reshape(my_inp,(1,5))
    
    #cd, cl
    out=model.predict([my_inp])
    out=out*np.asarray([0.25,0.9])
                    
    pred_cl=out[0,1]
    pred_cd=out[0,0]
    pred_clcd=pred_cl/pred_cd
    
    print ('Pred_cl:', pred_cl, pred_cd, pred_clcd)
    if(pred_clcd > tar_clcd):
        #e=np.sqrt(((tar_clcd - pred_clcd) ** 2).mean())
        e=0
    else:
        e=np.sqrt(((tar_clcd - pred_clcd) ** 2).mean())
    print ('mse:', e)
    
    error.append(e)
    
    my_counter = my_counter +1
    print ('Iter:', my_counter)
       
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(x,y,'r',label='true')
    plt.ylim([-0.5,0.5])
    plt.savefig('./opt_plot/%s.png'%my_counter,format='png')
    plt.close()
       
    
        
    return  e

#base foil name
#2412, 4510, 0014    
fn='naca0014'
    
idx=np.argwhere(name=='%s'%fn)

p1=mypara[idx[0][0],:]/scaler

 
print('Starting loss = {}'.format(loss(p1)))
print('Intial foil = %s' %name[idx[0]])

mylimit=((0,1.1),(0,1.1),(0.2,1.1))
#mylimit=((0,6),(0,6),(6,32))
res = minimize(loss, x0=p1, method = 'L-BFGS-B', bounds=mylimit, \
               options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
                                 'eps': 0.1, 'maxfun': 100, \
                                 'maxiter': 50, 'maxls': 100})
    
print('Ending loss = {}'.format(loss(res.x)))

fp=open('final_%s.dat'%fn,'w')
x,y=get_coord(res.x)
for i in range(len(x)):
    fp.write('%f %f 0.00\n'%(x[i],y[i]))
fp.close()

fp=open('resx_%s.dat'%fn,'w')
fp.write('%s'%res.x)
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
plt.savefig('./opti_%s.png'%fn,bbox_inches='tight',dpi=300)
plt.close()
