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


# read data from below dir...
#no use loop
## Time       	Cm           	Cd           	Cl           	Cl(f)        	Cl(r)
for jj in range(1):

    cfd=np.loadtxt('./case_2d_turb_2d_testing/testing/Re_120000/postProcessing/forceCoeffs/1/forceCoeffs.dat', skiprows=10)
    ml=np.loadtxt('./case_ml/Re_120000/postProcessing/forceCoeffs/147.2/forceCoeffs.dat', skiprows=10)
        
st=240  
idx1=np.argwhere(cfd[:,0]==130)[0][0]
idx2=np.argwhere(cfd[:,0]==150)[0][0]

mlx=np.concatenate((ml[:,0],ml[-1:,0]+0.2),axis=0)
mly1=np.concatenate((ml[:,3],ml[0:1,3]),axis=0)
mly2=np.concatenate((ml[:,2],ml[0:1,2]),axis=0)

for jj in range(1):
    
    #cl
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(cfd[idx1:idx2+1,0],cfd[idx1:idx2+1,3],'-o',lw=2,label='CFD')
    plt.plot(mlx,mly1,'-', lw=3,label='ML')
    plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.75), ncol=1, frameon=True, fancybox=False, shadow=False)
    plt.xlabel('time (s)',fontsize=20)
    plt.ylabel('Cl',fontsize=20)  
    #plt.axis('off')
    #plt.xlim([15000,30000])
    #plt.ylim([0.686,0.692])
    plt.tight_layout()
    plt.savefig('./plots/cl_re120000.png', bbox_inches='tight',dpi=100)
    plt.show()    
       
        
    #cl
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(cfd[idx1:idx2+1,0],cfd[idx1:idx2+1,2],'-o',lw=2,label='CFD')
    plt.plot(mlx,mly2,'-', lw=3,label='ML')
    plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.75), ncol=1, frameon=True, fancybox=False, shadow=False)
    plt.xlabel('time (s)',fontsize=20)
    plt.ylabel('Cd',fontsize=20)  
    #plt.axis('off')
    #plt.xlim([15000,30000])
    #plt.ylim([0.686,0.692])
    plt.tight_layout()
    plt.savefig('./plots/cd_re120000.png', bbox_inches='tight',dpi=100)
    plt.show()    
          
    
    
    
    
    
    
    '''fig, ax1 = plt.subplots(figsize=(6,5),dpi=100)
    
    ax1.set_xlabel('time (s)',fontsize=20)
    ax1.set_ylabel('Cl-CFD',fontsize=20)
    ax1.plot(les[st:,0],les[st:,3],'-o',lw=2,label='CFD')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cl-MLP',fontsize=20)
    ax2.plot([26.4, 26.8, 27.3, 27.8, 28.2],[0.61,0.629,0.7,0.63,0.62],'or',label='MLP')
    fig.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.75), ncol=1, frameon=True, fancybox=False, shadow=False)
    fig.tight_layout()
    fig.savefig('./plot/ts_%04d.png', bbox_inches='tight',dpi=100)
    fig.show()  '''
        
        
        
        
        

        
