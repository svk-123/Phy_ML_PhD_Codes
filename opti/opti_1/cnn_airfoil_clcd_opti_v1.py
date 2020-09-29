#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

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

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

# ref:[data,name]
path='./data_file/'
data_file='data_clcd.pkl'

with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)
print result1[-1:]
clcd=result1[1]    
name_clcd=result1[2]
clcd=np.asarray(clcd)
del result1

path='./data_file/'
data_file='param_216_16.pkl'

with open(path + data_file, 'rb') as infile:
    result2 = pickle.load(infile)
para=result2[0]    
name_para=result2[1]
del result2

mypara=[]
for i in range(len(name_clcd)):
    np.argwhere(name_clcd[i]==name_para)
    mypara.append(para[0][i])   
mypara=np.asarray(mypara)

# load model para to coord
model_pc=load_model('./selected_model/p16/model_cnn_2950_0.000013_0.000176.hdf5')  
get_out_1c= K.function([model_pc.layers[12].input],
                                  [model_pc.layers[15].output])

#load model clcd
model_clcd=load_model('./selected_model/model_cnn_275_0.000001_0.002375.hdf5') 

para_mm=[]
para_max=[]

for i in range(16):
    para_mm.append([mypara[:,i].min(),mypara[:,i].max()])
    para_max.append(mypara[:,i].max())
para_mm=np.asarray(para_mm)
para_max=np.asarray(para_max)    
para_mm[:,0]=para_mm[:,0]/(para_max+1e-6)
para_mm[:,1]=para_mm[:,1]/(para_max+1e-6)

bnds=tuple(para_mm)

global tar_cl
tar_cl=1.2

global my_counter
my_counter =0

def loss(para):
    
    para=para*para_max
    para=np.reshape(para,(1,16))
    c1 = get_out_1c([para])[0][0,:]
    c1=c1*0.25
    c1[35]=c1[0]
    
    xx=np.loadtxt('./xx.txt')
    
    co_x=np.concatenate((xx[::-1][:,None],xx[:,None]))
    co_c1=np.concatenate((c1[:35][::-1][:,None],c1[35:,None]))
    
    figure=plt.figure(figsize=(2,2))
    plt0, =plt.plot(co_x,co_c1,'k',linewidth=0.5)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.18,0.18)    
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.savefig('./plot/tmp.eps', format='eps')
    plt.close() 
    
    img = io.imread('./plot/tmp.eps', as_grey=True)  # load the image as grayscale
    img = util.invert(img)
    
    img=np.asarray(img)
    val_inp=np.reshape(img,(1,144,144,1))  
    out=model_clcd.predict([val_inp])
    pred_cl=out[0,0]*1.6
    print pred_cl
    print my_counter
    global my_counter
    my_counter = my_counter +1
    
    return max(0, (tar_cl-pred_cl))
     

p1=mypara[50,:]
p1=p1/(para_max+1e-6)
    
print('Starting loss = {}'.format(loss(p1)))

res = minimize(loss, x0=p1, method = 'L-BFGS-B', bounds=bnds, \
               options={'disp': True, 'maxcor':1, 'ftol': 1e-16, \
                                 'eps': 1e-4, 'maxfun': 1, \
                                 'maxiter': 20, 'maxls': 1})
print('Ending loss = {}'.format(loss(res.x)))


f1= get_out_1c([np.reshape(p1*para_max,(1,16))])[0][0,:]
f1[35]=f1[0]
f1=f1*0.25
f2= get_out_1c([np.reshape(res.x*para_max,(1,16))])[0][0,:]
f2[35]=f2[0]
f2=f2*0.25

xx=np.loadtxt('./xx.txt')
co_x=np.concatenate((xx[::-1][:,None],xx[:,None]))
co_f1=np.concatenate((f1[:35][::-1][:,None],f1[35:,None]))
co_f2=np.concatenate((f2[:35][::-1][:,None],f2[35:,None]))

#figure
figure=plt.figure(figsize=(6,5))
plt.plot(co_x,co_f1,'r',label='base')
plt.plot(co_x,co_f2,'g',label='optimized')
plt.xlim(-0.02,1.02)
plt.ylim(-0.18,0.18) 
plt.legend()
plt.savefig('opti_1.eps',format='eps')
plt.show()

