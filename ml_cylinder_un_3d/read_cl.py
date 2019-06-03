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

    xx=np.loadtxt('./postProcessing/forceCoeffs/0/forceCoeffs.dat', skiprows=10)
    
st=0
for jj in range(1):
    
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx[st:,0],xx[st:,3],'-o',lw=2,label='cl')
    #plt.plot(range(len(rans)),rans[:,3],'b',lw=3,label='rans')
    plt.legend(fontsize=20)
    plt.xlabel('time (s)',fontsize=20)
    plt.ylabel('Cl',fontsize=20)  
    #plt.axis('off')
    #plt.xlim([15000,30000])
    #plt.ylim([-0.05,0.05])
    plt.tight_layout()
    #plt.savefig('./plot/naca_10000_18.png', bbox_inches='tight',dpi=100)
    plt.show()    
       
        
        
        
        
        
        
        

        
