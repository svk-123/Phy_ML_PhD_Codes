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
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

from naca import naca4

# ref:[data,name]
#load airfoil para
path='./data_file/'
data_file='naca4_lam_clcd_3para.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile,encoding='bytes')

cd=result1[1]
cl=result1[2]
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
model=load_model('./selected_model/turb_3para_st_6x30/final_sf.hdf5')  

global scaler
scaler=np.array([6,6,30])

global tar_cl
global pred_cl
global reno
global aoa
global init_cl

tar_cl=np.asarray([0.627,0.773,0.898,0.930])
pred_cl=np.asarray([0,0,0,0])
init_cl=np.asarray([0,0,0,0])
reno=np.asarray([10000,20000,40000,50000])/100000.
aoa=np.asarray([4])/14.

reno=np.asarray(reno)
aoa=np.asarray(aoa)

global my_counter
my_counter =0


fp=open('conv.dat','w+')


def get_coord(p2):
    x,y=naca4(p2*scaler,100)
    return (x,y)

def loss(para):
       
    global my_counter  
    global pred_cl
    global init_cl
    para=para

    x,y=get_coord(para)


    inp_aoa=np.repeat(aoa, len(reno)) 
    inp_para=np.repeat(para[:,None],len(reno),axis=1).transpose() 
    
    my_inp=np.concatenate((reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
    

    #cd, cl
    out=model.predict([my_inp])
    out=out*np.asarray([0.25,0.9])
                
    pred_cl=out[:,1]
    
    print ('Pred_cl:', pred_cl)

    e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
    
    print ('mse:', e)
    
    fp.write('%s %s %s\n'%(my_counter,e,pred_cl))    
    if(my_counter == 0):
        init_cl=pred_cl
    my_counter = my_counter +1
    print ('Iter:', my_counter)
       
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(x,y,'r',label='true')
    plt.ylim([-0.5,0.5])
    plt.savefig('./opt_plot/%s.png'%my_counter,format='png')
    plt.close()
       
            
    return  e
     
idx=np.argwhere(name=='naca0014')
#idx=np.argwhere(name=='naca4510')
#naca4510
p1=mypara[idx[0][0],:]/scaler



   
print('Starting loss = {}'.format(loss(p1)))
print('Intial foil = %s' %name[idx[0]])

mylimit=((0,1.1),(0,1.1),(0.2,1.1))
res = minimize(loss, x0=p1, method = 'L-BFGS-B', bounds=mylimit, \
               options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
                                 'eps': 1e-2, 'maxfun': 100, \
                                 'maxiter': 50, 'maxls': 100})
   
 
    
    
print('Ending loss = {}'.format(loss(res.x)))


fp=open('final.dat','w')
x,y=get_coord(res.x)
for i in range(len(x)):
    fp.write('%f %f 0.00\n'%(x[i],y[i]))
fp.close()

fp=open('resx.dat','w')
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
plt.savefig('./opti_0014.png',format='png',bbox_inches='tight',dpi=300)
plt.close()


plt.figure(figsize=(6,5),dpi=100)
plt.plot(reno*100000,tar_cl,'-ok',lw=3,label='Target')
plt.plot(reno*100000,init_cl,'-ob',lw=3,label='Base')
plt.plot(reno*100000,pred_cl,'-og',lw=3,label='Optimized')
plt.legend(fontsize=14)
plt.xlabel('Reynolds No',fontsize=16)
plt.ylabel('Cl',fontsize=16)
plt.savefig('./line_0014.png',format='png',bbox_inches='tight',dpi=300)
plt.close()