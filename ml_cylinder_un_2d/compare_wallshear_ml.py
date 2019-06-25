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
    
        #load shearstress 
        tu1=[]
        tu2=[]
        
        with open('./case_2d_turb_2d_testing/testing/Re_120000' + '/143.4/wallShearStress', 'r') as infile:
            
            data0=infile.readlines()
            ln=0
            for line in data0:
                if 'walls' in line:
                    idx=ln
                ln=ln+1    
            num=int(data0[idx+4])
            
            for line in data0[idx+6:idx+6+num]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c = (item.strip() for item in line.split(' ', 3))
                tu1.append(a), tu2.append(b)
        tu1 = np.array(map(float, tu1))
        tu2 = np.array(map(float, tu2))
        tu3=np.sqrt(tu1**2 + tu2**2) 
          
        
        #load shearstress 
        tu1p=[]
        tu2p=[]
        
        with open('./case_ml/Re_120000' + '/143.4/wallShearStress', 'r') as infile:
            
            data0=infile.readlines()
            ln=0
            for line in data0:
                if 'walls' in line:
                    idx=ln
                ln=ln+1    
            num=int(data0[idx+4])
            
            for line in data0[idx+6:idx+6+num]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c = (item.strip() for item in line.split(' ', 3))
                tu1p.append(a), tu2p.append(b)
        tu1p = np.array(map(float, tu1p))
        tu2p = np.array(map(float, tu2p))
        tu3p=np.sqrt(tu1p**2 + tu2p**2)     
          
plt.figure(figsize=(6,5),dpi=100)
plt.plot(np.linspace(0,360,149),tu3,'-b',lw=2,label='CFD')
plt.plot(np.linspace(0,360,149),tu3p,'-r',lw=2,label='ML')
plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.75), ncol=1, frameon=True, fancybox=False, shadow=False)
plt.xlabel('theta',fontsize=20)
plt.ylabel('mag(tau)',fontsize=20)  
plt.savefig('./plots/tau_re120000.png', bbox_inches='tight',dpi=100)
plt.show()          
        
        
        

        
