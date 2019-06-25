#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:49:29 2017

This code process OF data and exports as .pkl to prepData file
for TBNN. prepData reads .pkl and process further

@author: vino
"""
# imports
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy import interpolate
from os import listdir
from os.path import isfile,isdir, join
import pickle


import keras
from keras.models import load_model
import shutil

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""
import matplotlib
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

## ml - prediction
indir = './case_ml/Re_120000/postProcessing/sampleDict'

fname_1 = [f for f in listdir(indir) if isdir(join(indir, f))]
fname_1.sort()
fname_1=np.asarray(fname_1)
fname_1=fname_1[1:] #remove 0
fname_1.sort()

# t ,p , u, v
ml=[]
for i in range(len(fname_1)):
    x1=np.loadtxt(indir +'/%s/data_p.csv'%fname_1[i],delimiter=',',skiprows=1)
    x2=np.loadtxt(indir +'/%s/data_U.csv'%fname_1[i],delimiter=',',skiprows=1)
    ml.append([np.float(fname_1[i]),x1[3],x2[3],x2[4]])    
ml=np.asarray(ml)
        
## CFD
indir = './case_2d_turb_2d_testing/testing/Re_120000/postProcessing/sampleDict'

fname_1 = [f for f in listdir(indir) if isdir(join(indir, f))]
fname_1.sort()
fname_1=np.asarray(fname_1)
fname_1=fname_1[2:] #remove 0
fname_1.sort()

# t ,p , u, v
cfd=[]
for i in range(len(fname_1)):
    x1=np.loadtxt(indir +'/%s/data_p.csv'%fname_1[i],delimiter=',',skiprows=1)
    x2=np.loadtxt(indir +'/%s/data_U.csv'%fname_1[i],delimiter=',',skiprows=1)
    cfd.append([np.float(fname_1[i]),x1[3],x2[3],x2[4]]) 
cfd=np.asarray(cfd)
        
#p
plt.figure(figsize=(6,5),dpi=100)
plt.plot(cfd[:,0],cfd[:,1],'-o',lw=2,label='CFD-p')
plt.plot(ml[:,0],ml[:,1],'-', lw=3,label='ML-p')
plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.75), ncol=1, frameon=True, fancybox=False, shadow=False)
plt.xlabel('time (s)',fontsize=20)
plt.ylabel('p',fontsize=20)  
plt.savefig('./plots/p_re120000.png', bbox_inches='tight',dpi=100)
plt.show()    

#u
plt.figure(figsize=(6,5),dpi=100)
plt.plot(cfd[:,0],cfd[:,2],'-o',lw=2,label='CFD-u')
plt.plot(ml[:,0],ml[:,2],'-', lw=3,label='ML-u')
plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.75), ncol=1, frameon=True, fancybox=False, shadow=False)
plt.xlabel('time (s)',fontsize=20)
plt.ylabel('u',fontsize=20)  
plt.savefig('./plots/u_re120000.png', bbox_inches='tight',dpi=100)
plt.show() 


#v
plt.figure(figsize=(6,5),dpi=100)
plt.plot(cfd[:,0],cfd[:,3],'-o',lw=2,label='CFD-v')
plt.plot(ml[:,0],ml[:,3],'-', lw=3,label='ML-v')
plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.75), ncol=1, frameon=True, fancybox=False, shadow=False)
plt.xlabel('time (s)',fontsize=20)
plt.ylabel('v',fontsize=20)  
plt.savefig('./plots/v_re120000.png', bbox_inches='tight',dpi=100)
plt.show() 



