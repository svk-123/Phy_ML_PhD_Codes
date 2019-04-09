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

    les=np.loadtxt('./les/les_s805_Re_1e6_aoa_14/postProcessing/forceCoeffs/0/forceCoeffs.dat', skiprows=10)
    #rans=np.loadtxt('./rans/rans_naca_0012_aoa_6/postProcessing/forceCoeffs/0/forceCoeffs.dat', skiprows=10)
    
    
    
for jj in range(1):
    
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(range(len(les[:,3][100000:])),les[:,3][100000:],'r',lw=3,label='les')
    #plt.plot(range(len(rans)),rans[:,3],'b',lw=3,label='rans')
    plt.legend(fontsize=20)
    plt.xlabel('X',fontsize=20)
    plt.ylabel('Y',fontsize=20)  
    #plt.axis('off')
    #plt.xlim([50000,200000])
    #plt.ylim([1.2,1.8])
    plt.tight_layout()
    #plt.savefig('./plot/ts_%04d.png'%(k), bbox_inches='tight',dpi=100)
    plt.show()    
       
        
        
        
        
        
        
        

        