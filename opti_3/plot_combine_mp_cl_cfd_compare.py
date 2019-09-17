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

path='./result_paper_v2/mp_1_tanh/'

##for mp-1-relu
#name=['naca0016','naca0028','naca0222','naca3416','naca5024','naca5126']
#tar_cl=[0.627, 0.773, 0.898, 0.93]
#reno=[10000,20000,40000,50000]
#
#n0016=[0.586, 0.793, 0.915, 0.939]
#n0028=[0.626, 0.801, 0.920, 0.948]
#n0222=[0.557, 0.799, 0.914, 0.937]
#n3416=[0.574, 0.799, 0.911, 0.933]
#n5024=[0.626, 0.799, 0.916, 0.942]
#n5126=[0.537, 0.827, 0.937, 0.962]

#for mp-1-tanh
name=['naca0016','naca3416','naca0226','naca2130','naca4524']
tar_cl=[0.627, 0.773, 0.898, 0.93]
reno=[10000,20000,40000,50000]

n0016=[0.596, 0.791, 0.920, 0.948]
n3416=[0.598, 0.791, 0.921, 0.950]
n0226=[0.509, 0.786, 0.897, 0.923]
n2130=[0.531, 0.802, 0.911, 0.933]
n4524=[0.581, 0.788, 0.917, 0.945]


##for mp-2-relu
#name=['naca5014','naca5008','naca3012','naca2030','naca2010','naca1416']


##name=foil
#c=['b','b','y','c','r','m']
#base=[]
#opti=[]
#
#for ii in range(5):
#    tmp=np.loadtxt(path + 'base_%s.dat'%name[ii])
#    base.append(tmp)
#    tmp=np.loadtxt(path + 'final_%s.dat'%name[ii])
#    opti.append(tmp)
#
#plt.figure(figsize=(6,5),dpi=100)
#for ii in [0,1,2,3,4]:
#    if(ii==0):
#        #plt.plot(base[ii][:,0],base[ii][:,1],'--k',lw=1,label='Base Shapes')
#        plt.plot(opti[ii][:,0],opti[ii][:,1],'g',lw=2,label='Optimized Shape-%d'%ii)
#    else:
#        #plt.plot(base[ii][:,0],base[ii][:,1],'--k',lw=1)
#        plt.plot(opti[ii][:,0],opti[ii][:,1],'-%s'%c[ii],lw=2,label='Optimized Shape-%d'%ii)    
#       
#        
#plt.legend(loc="upper left", bbox_to_anchor=[1, 1], ncol=1, fontsize=14, \
#           frameon=False, shadow=False, fancybox=False,title='')
#plt.xlim([-0.05,1.05])
#plt.ylim([-0.25,0.25])
#plt.xlabel('X',fontsize=16)
#plt.ylabel('Y',fontsize=16)
#plt.savefig(path+'report_relu.png',bbox_inches='tight',dpi=300)
#plt.show()
#plt.close()
#
##CFD comp
#plt.figure(figsize=(6,5),dpi=100)
#plt.plot(reno,tar_cl,'k',marker='o',mfc='k',mec='k',ms=12,lw=1,markevery=1,label='Tar_Cl')
#plt.plot(reno,n0016,'g',marker='o',mfc='g',mec='g',ms=12,lw=1,markevery=1,label='Optimized Cl for shape-0 (CFD)')
#plt.plot(reno,n0028,'b',marker='o',mfc='b',mec='b',ms=12,lw=1,markevery=1,label='Optimized Cl for shape-1 (CFD)')
#plt.plot(reno,n0222,'y',marker='o',mfc='y',mec='y',ms=12,lw=1,markevery=1,label='Optimized Cl for shape-2 (CFD)')
#plt.plot(reno,n3416,'c',marker='o',mfc='c',mec='c',ms=12,lw=1,markevery=1,label='Optimized Cl for shape-3 (CFD)')
#plt.plot(reno,n5024,'r',marker='o',mfc='r',mec='r',ms=12,lw=1,markevery=1,label='Optimized Cl for shape-4 (CFD)')
#plt.plot(reno,n5126,'r',marker='o',mfc='m',mec='m',ms=12,lw=1,markevery=1,label='Optimized Cl for shape-5 (CFD)')
#plt.legend(loc="upper left", bbox_to_anchor=[1, 1], ncol=1, fontsize=14, \
#           frameon=False, shadow=False, fancybox=False,title='')
#plt.xlim([5000,55000])
#plt.ylim([0.5,1.1])
#plt.xlabel('X',fontsize=16)
#plt.ylabel('Y',fontsize=16)
#plt.savefig(path+'report_relu_cl_cfd.png',bbox_inches='tight',dpi=300)
#plt.show()
#plt.close()


#NN pred comp (conv)
c=['g','b','y','c','r','m']
plt.figure(figsize=(6,5),dpi=100)
plt.plot(reno,tar_cl,'k',marker='o',mfc='k',mec='k',ms=12,lw=1,markevery=1,label='Tar_Cl')
for ii in [0,1,2,3,4]:
    tmp=np.loadtxt(path + 'conv_%s.dat'%name[ii],skiprows=2)
    plt.plot(reno,tmp[-1:,2:6][0],'%s'%c[ii],marker='o',mfc='%s'%c[ii],mec='%s'%c[ii],ms=12,lw=1,markevery=1,label='Optimized Cl for shape-%s (NN)'%ii)
plt.legend(loc="upper left", bbox_to_anchor=[1, 1], ncol=1, fontsize=14, \
           frameon=False, shadow=False, fancybox=False,title='')
plt.xlim([5000,55000])
plt.ylim([0.5,1.1])
plt.xlabel('X',fontsize=16)
plt.ylabel('Y',fontsize=16)
plt.savefig(path+'report_relu_cl_nn.png',bbox_inches='tight',dpi=300)
plt.show()
plt.close()








#
##mp-1-relu
#error=[]
#
#tmp=np.asarray(tar_cl)-np.asarray(n0016)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)
# 
#tmp=np.asarray(tar_cl)-np.asarray(n0028)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)
#
#tmp=np.asarray(tar_cl)-np.asarray(n0222)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)
#
#tmp=np.asarray(tar_cl)-np.asarray(n3416)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)
#
#tmp=np.asarray(tar_cl)-np.asarray(n5024)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)
#
#tmp=np.asarray(tar_cl)-np.asarray(n5126)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)



##mp-1-tanh
#error=[]
#
#tmp=np.asarray(tar_cl)-np.asarray(n0016)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)
# 
#tmp=np.asarray(tar_cl)-np.asarray(n3416)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)
#
#tmp=np.asarray(tar_cl)-np.asarray(n0226)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)
#
#tmp=np.asarray(tar_cl)-np.asarray(n2130)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)
#
#tmp=np.asarray(tar_cl)-np.asarray(n4524)
#error.append((LA.norm(tmp)/LA.norm(tar_cl))*100)