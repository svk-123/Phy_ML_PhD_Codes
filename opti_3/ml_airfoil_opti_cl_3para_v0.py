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

from naca import naca4

# ref:[data,name]
#load airfoil para
path='./data_file/'
data_file='naca4_lam_clcd_3para.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)

cd=result1[1]
cl=result1[2]
myreno=result1[3]
myaoa=result1[4]
mypara=result1[5]
name=result1[6]

mypara=np.asarray(mypara)
name=np.asarray(name)

#fp=open('cl_data.txt','w')
#for i in range(len(cl)):
#    fp.write('%s %f %f %f \n'%(name[i],myreno[i],myaoa[i],cl[i]))
#    
#fp.close()
    

#load model para: airfoil coord from parameters
#scaler used in cl pred network
# reno=10000
#aoa=14
#para=np.array([6,6,30])
#cl=0.8
#cd=0.25
model=load_model('./selected_model/3para_6x30/model_sf_1500_0.00041091_0.00041618.hdf5')  
global scaler
scaler=np.array([6,6,30])

global tar_cl
global reno
global aoa

tar_cl=0.6
reno=np.asarray([1200])/10000.
aoa=np.asarray([6])/14.

reno=np.asarray(reno)
aoa=np.asarray(aoa)

global my_counter
my_counter =0

def get_coord(p2):
    x,y=naca4(p2*scaler,100)
    return (x,y)

def loss(para):
       
    global my_counter  
    para=para

    x,y=get_coord(para)

    my_inp=np.concatenate((reno,aoa,para),axis=0)
    my_inp=np.reshape(my_inp,(1,5))
    #cd, cl
    out=model.predict([my_inp])
    out=out*np.asarray([0.25,0.8])
                
    pred_cl=out[0,1]
    
    print ('Pred_cl:', pred_cl)
    e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
    print ('mse:', e)
    
    my_counter = my_counter +1
    print ('Iter:', my_counter)
       
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(x,y,'r',label='true')
    plt.ylim([-0.5,0.5])
    plt.savefig('./opt_plot/%s.png'%my_counter,format='png')
    plt.close()
        
    return  e
     


p1=mypara[10000,:]/scaler
    
print('Starting loss = {}'.format(loss(p1)))

res = minimize(loss, x0=p1, method = 'L-BFGS-B', \
               options={'disp': True, 'maxcor':20, 'ftol': 1e-3, \
                                 'eps': 1e-1, 'maxfun': 20, \
                                 'maxiter': 100, 'maxls': 20})
print('Ending loss = {}'.format(loss(res.x)))

'''
get_coord= K.function([model_para.layers[16].input],[model_para.layers[19].output])

my_para=np.reshape(res.x,(1,16))*scaler
# requires input shape (1,16)
c1 = get_coord([my_para])[0][0,:]
co=np.zeros((69,2))
co[0:35,0]=xx[::-1]
co[0:35,1]=c1[:35]
co[35:69,0]=xx[1:]
co[35:69,1]=c1[36:]

base_para=np.reshape(p1,(1,16))*scaler
co_b=co.copy()
c2 = get_coord([base_para])[0][0,:]
co_b[0:35,0]=xx[::-1]
co_b[0:35,1]=c2[:35]
co_b[35:69,0]=xx[1:]
co_b[35:69,1]=c2[36:]

plt.figure(figsize=(6,5),dpi=100)
plt.plot(co_b[:,0],co_b[:,1]*0.25,'r',label='base')
plt.plot(co[:,0],co[:,1]*0.25,'g',label='optimized')
plt.ylim([-0.3,0.3])
plt.legend()
plt.savefig('./gen_sp.png',format='png')
plt.close()
'''