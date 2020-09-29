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

a='mp_1_tanh_5p/'
path_out='./paper_plots/' + a
path='./result_paper_v7/' + a

##for mp-1-relu
#name=['naca0016','naca0028','naca0222','naca3416','naca4414','naca4524','naca5024']
name=[]
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and 'opti' in i:
        name.append(i.split('_')[1].split('.')[0])
        print i

#name=foil
c=['darkorange','lime','pink','purple','y','g','b','c','r','m','peru','gold','olive','salmon','brown',\
   'g','b','y','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown']

base=[]
opti=[]

for ii in range(len(name)):
    tmp=np.loadtxt(path + 'base_%s.dat'%name[ii])
    base.append(tmp)
    tmp=np.loadtxt(path + 'final_%s.dat'%name[ii])
    opti.append(tmp)

plt.figure(figsize=(6,5),dpi=100)
for ii in range(len(name)):
    if(ii==0):
        plt.plot(base[ii][:,0],base[ii][:,1],'%s'%c[ii],lw=1,label='NACA %s'%name[ii].split('naca')[1])
        
    else:
        plt.plot(base[ii][:,0],base[ii][:,1],'%s'%c[ii],lw=1,label='NACA %s'%name[ii].split('naca')[1])
  
        
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, \
           frameon=False, shadow=False, fancybox=False,title='')
plt.text(1.1,0.2,'Base Airfoil',fontsize=16)
plt.xlim([-0.05,1.05])
plt.ylim([-0.25,0.25])
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
plt.figtext(0.50, -0.07, '(a)', wrap=True, horizontalalignment='center', fontsize=24) 
plt.savefig(path_out+'base.tiff',format='tiff',bbox_inches='tight',dpi=300)
plt.show()
plt.close()



plt.figure(figsize=(6,5),dpi=100)
a=[5,4,3,2,4,3,2,4,2,2]
b=[2,3,2,3,2,3,2,3,2,3,2,3]
for ii in range(len(name)):
    if(ii==0):
        #plt.plot(base[ii][:,0],base[ii][:,1],'gray',lw=1,label='Base Shapes')
        plt.plot(opti[ii][:,0],opti[ii][:,1],'g',lw=3,dashes=[1, 2],label='NACA %s'%name[ii].split('naca')[1])
        
    else:
        #plt.plot(base[ii][:,0],base[ii][:,1],'gray',lw=1)
        plt.plot(opti[ii][:,0],opti[ii][:,1],'%s'%c[ii],lw=3,dashes=[a[ii], b[ii]],label='NACA %s'%name[ii].split('naca')[1])    
        
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.8], ncol=1, fontsize=14, \
           frameon=False, shadow=False, fancybox=False,title='')
plt.text(1.1,0.2,'Optimized Shape',fontsize=16)
plt.text(1.1,0.16,'for base Airfoil',fontsize=16)
plt.xlim([-0.05,1.05])
plt.ylim([-0.25,0.25])
plt.figtext(0.50, -0.07, '(b)', wrap=True, horizontalalignment='center', fontsize=24) 
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
plt.savefig(path_out+'optimized.tiff',format='tiff',bbox_inches='tight',dpi=300)
plt.show()
plt.close()



conv=[]

for ii in range(len(name)):
    tmp=np.loadtxt(path + 'conv_%s.dat'%name[ii],skiprows=2)
    conv.append(tmp)

N=50
plt.figure(figsize=(6,5),dpi=100)
for ii in range(len(name)):
    tmp=np.sqrt(conv[ii][:N,2]**2 + conv[ii][:N,3]**2 + conv[ii][:N,4]**2)
    
    plt.plot(conv[ii][:N,0], tmp,'%s'%c[ii],lw=3,label='NACA %s'%name[ii].split('naca')[1])
ymax=tmp.max()    
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.8], ncol=1, fontsize=14, \
           frameon=False, shadow=False, fancybox=False,title='')
plt.text(N+3,ymax*0.94,'Convergence',fontsize=16)
plt.text(N+3,ymax*0.94-0.15,'for base Airfoil',fontsize=16)
plt.figtext(0.50, -0.07, '(d)', wrap=True, horizontalalignment='center', fontsize=24) 
#plt.xlim([-0.05,30])
#plt.ylim([-0.05,1.1])
plt.xlabel('Iter',fontsize=20)
plt.ylabel('$||C_L||_2$',fontsize=20)
plt.savefig(path_out+'conv.tiff',format='tiff',bbox_inches='tight',dpi=300)
plt.show()
plt.close()



tar_cl=[]
pred_cl=[]
init_cl=[]

tar_cd=[]
pred_cd=[]
init_cd=[]

for i in range(1):
    with open(path +'resx_%s.dat'%name[i], 'r') as infile:
        data0=infile.readlines()
       
    for j in range(len(data0)):
        if 'tar-cl' in data0[j]:
            tar_cl.append(data0[j])
        if 'tar-cd' in data0[j]:
            tar_cd.append(data0[j])  
            
tar_cl= np.asarray(tar_cl[0].split('=')[1].replace('[','').replace(']','').split())
tar_cd= np.asarray(tar_cd[0].split('=')[1].replace('[','').replace(']','').split())            

tar_cl=tar_cl.astype(np.float)
tar_cd=tar_cd.astype(np.float)

for i in range(len(name)):
    with open(path +'resx_%s.dat'%name[i], 'r') as infile:
        data0=infile.readlines()
       
    for j in range(len(data0)):
        if 'pred-cl' in data0[j]:
            pred_cl.append(data0[j])
        if 'pred-cd' in data0[j]:
            pred_cd.append(data0[j])  
        if 'init-cl' in data0[j]:
            init_cl.append(data0[j])
        if 'init-cd' in data0[j]:
            init_cd.append(data0[j]) 

for i in range(len(name)):            
    pred_cl[i]= pred_cl[i].split('=')[1].replace('[','').replace(']','').split()
    pred_cd[i]= pred_cd[i].split('=')[1].replace('[','').replace(']','').split()
    init_cl[i]= init_cl[i].split('=')[1].replace('[','').replace(']','').split()
    init_cd[i]= init_cd[i].split('=')[1].replace('[','').replace(']','').split()

pred_cl=np.asarray(pred_cl)
pred_cd=np.asarray(pred_cd)
init_cl=np.asarray(init_cl)
init_cd=np.asarray(init_cd)

pred_cl=pred_cl.astype(np.float)
pred_cd=pred_cd.astype(np.float)
init_cl=init_cl.astype(np.float)
init_cd=init_cd.astype(np.float)

a=[5,4,3,2,4,3,2,4,2,2]
b=[2,3,2,3,2,3,2,3,2,3,2,3]
reno=[40000,50000,50000,50000,60000]
plt.figure(figsize=(6,5),dpi=100)

for ii in range(len(name)):
    plt.plot(reno,init_cl[ii],'%s'%c[ii],marker='s',lw=2,label='NACA %s'%name[ii].split('naca')[1])
for ii in range(len(name)):    
    plt.plot(reno,pred_cl[ii],'%s'%c[ii],marker='o',ms=8,lw=3,dashes=[a[ii], b[ii]],label='NACA %s'%name[ii].split('naca')[1])
for ii in range(1):
    plt.plot(reno,tar_cl,'gray',marker='o',ms=8,lw=0)      
        
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.8], ncol=2, fontsize=12, \
           frameon=False, shadow=False, fancybox=False,title='')
plt.text(60000*1.03,1.05,'Initial $C_L$',fontsize=16)
plt.text(60000*1.18,1.05,'Pred $C_L$',fontsize=16)
plt.xticks([40000,50000,60000]) 
plt.figtext(0.50, -0.07, '(c)', wrap=True, horizontalalignment='center', fontsize=24) 
#plt.xlim([-0.05,30])
#plt.ylim([-0.05,1.1])
plt.xlabel('Re',fontsize=20)
plt.ylabel('$C_L$',fontsize=20)
plt.savefig(path_out+'cl.tiff',format='tiff',bbox_inches='tight',dpi=300)
plt.show()
plt.close()
plt.figure(figsize=(6,5),dpi=100)

for ii in range(len(name)):
    plt.plot(reno,init_cd[ii],'%s'%c[ii],marker='s',lw=2,label='NACA %s'%name[ii].split('naca')[1])
for ii in range(len(name)):    
    plt.plot(reno,pred_cd[ii],'%s'%c[ii],marker='o',ms=8,lw=3,dashes=[a[ii], b[ii]],label='NACA %s'%name[ii].split('naca')[1])
           
   
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.8], ncol=2, fontsize=12, \
           frameon=False, shadow=False, fancybox=False,title='')
plt.text(60000*1.03,0.053,'Initial Cl',fontsize=16)
plt.text(60000*1.18,0.053,'Pred Cl',fontsize=16)
plt.xticks([40000,50000,60000]) 
#plt.xlim([-0.05,30])
#plt.ylim([-0.05,1.1])
plt.xlabel('Re',fontsize=20)
plt.ylabel('$C_D$',fontsize=20)
plt.savefig(path_out+'cd.tiff',format='tiff',bbox_inches='tight',dpi=300)
plt.show()
plt.close()