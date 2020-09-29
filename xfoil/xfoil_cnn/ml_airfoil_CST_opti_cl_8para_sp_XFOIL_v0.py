#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

with new parameters tanh_16_v1:

with new flow prediction network using v1.

@author: vinoth
"""
#based on parameters 
from __future__ import division
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

##Xfoil imports
from kulfan_to_coord import CST_shape
import argparse
from utils import mean_err
from simulation import evaluate,compute_coeff
####------------


import matplotlib
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 


# ref:[data,name]
folder = './opt_plot/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

cd=[]
cl=[]



#load airfoil para
name=[]
for i in range(1):
    with open('./model_cst_gen/foilname.dat', 'r') as infile:
        data0=infile.readlines()
    for j in range(len(data0)):
        name.append(data0[j].strip())
name=np.asarray(name)       

mypara=np.loadtxt('./model_cst_gen/cst.dat')
 
#nname=[]
#for i in range(len(name)):
#    nname.append(name[i].decode())
#name=np.asarray(nname)

#fp=open('cl_data_turb_gen_st.txt','w')
#for i in range(len(cl)):
#    fp.write('%s %f %f %f \n'%(name[i],myreno[i],myaoa[i],cl[i]))
#    
#fp.close()


global tar_cl
global init_cl
global reno
global aoa

tar_cl=1.0
init_cl=0
reno=50000.0
aoa=6.0

reno=np.asarray(reno)
aoa=np.asarray(aoa)

global my_counter
my_counter =0

global error
error=[]

global xx
xx=np.loadtxt('./model_cst_gen/2032c.dat')
xx=xx[:,0]

def get_coord(p2):
    
    airfoil_CST2 = CST_shape(p2[0:4], p2[4:8], 0, 100)
    coordinates = airfoil_CST2.airfoil_coor()
    x_coor = coordinates[0]
    y_coor = coordinates[1]
    x_coor=np.concatenate((x_coor,x_coor[0:1]),axis=0)
    y_coor=np.concatenate((y_coor,-y_coor[0:1]),axis=0)
    return (np.asarray([x_coor,y_coor]))

def loss(para):
       
    global my_counter  
    global init_cl
    global pred_cl
    mypara=para

    xy=get_coord(mypara)
                   
    _,pred_cl,pred_cd=evaluate(xy.transpose(),True)
    
    if(pred_cl > 0):
        e=pred_cd/pred_cl
    else:
        e=10.0    
    
#    if(pred_cl > tar_cl):
#        #e=np.sqrt(((tar_cl - pred_cl) ** 2))
#        e=0
#    else:
#        e=np.sqrt(((tar_cl - pred_cl) ** 2))
     
    
    fp.write('%s %s %s\n'%(my_counter,e,pred_cl))    
    if(my_counter == 0):
        init_cl=pred_cl
    my_counter = my_counter +1
    print ('Iter: %d, E: %f , Pred_cl: %f' %(my_counter,e,pred_cl/pred_cd))
    
     
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xy[0,:],xy[1,:],'r')
    plt.ylim([-0.5,0.5])
    plt.savefig('./opt_plot/%s.png'%my_counter,format='png')
    plt.close()
                  
    return  e
     

#base foil name
fn='naca0010'

idx=np.argwhere(name=='%s'%fn)

#scaled parameter
p1=mypara[idx[0][0],:]

#conv file
fp=open('./tmp1/conv_%s.dat'%fn,'w+')
 
print('Starting loss = {}'.format(loss(p1)))
print('Intial foil = %s' %name[idx[0]])

mylimit=((-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1))
#mylimit=((0,6),(0,6),(6,32))
res = minimize(loss, x0=p1, method = 'L-BFGS-B', bounds=mylimit, \
               options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
                                 'eps': 0.0002, 'maxfun': 100, \
                                 'maxiter': 100, 'maxls': 100})
    
#res = minimize(loss, x0=p1, method = 'L-BFGS-B', \
#               options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
#                                 'eps': 0.01, 'maxfun': 100, \
#                                 'maxiter': 50, 'maxls': 100})
    

print('Ending loss = {}'.format(loss(res.x)))
fp.close()


fp=open('./tmp1/final_%s.dat'%fn,'w')
x,y=get_coord(res.x)
for i in range(len(x)):
    fp.write('%f %f 0.00\n'%(x[i],y[i]))
fp.close()

fp=open('./tmp1/resx_%s.dat'%fn,'w')
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
plt.savefig('./tmp1/opti_%s.png'%fn,format='png',bbox_inches='tight',dpi=300)
plt.close()