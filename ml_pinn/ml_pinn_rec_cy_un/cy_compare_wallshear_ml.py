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



fname_1=['case_ml']

fname_1=np.asarray(fname_1)
fname_1.sort()


# read data from below dir...
path='./'
#path='/home/vino/ml_test/from_nscc_26_dec_2018/foam_run/case_naca_lam'
indir = path



#np.random.seed(1234)
#np.random.shuffle(fname)

fname_2=[]
for i in range(len(fname_1)):
    dir2=indir + '/%s'%fname_1[i]
    tmp=[f for f in listdir(dir2) if isdir(join(dir2, f))]
    fname_2.append(tmp)
    
fname_2=np.asarray(fname_2)    
fname_2.sort()

tmp=[]
foil=[]
for i in range(len(fname_1)):
    for j in range(len(fname_2[i])):
        tmp.append(fname_2[i][j])
        foil.append(fname_2[i][j].split('_')[0])
tmp=np.asarray(tmp)    
foil=np.asarray(foil)

mylabel=['CFD','PINN']


tt=[142.2,143.8,145.4]

T=[]
for kk in [2]:
    for jj in range(2):
        
            #load shearstress 
            tu1=[]
            tu2=[]
    
            with open('./case_ml/%s/%s/wallShearStress'%(tmp[jj],tt[kk]), 'r') as infile:
                
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
            T.append(-2*0.01*tu1)  
        
   
c=['g','b','y','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown']          
plt.figure(figsize=(6,5),dpi=100)
plt.plot(np.linspace(0,180,100),T[0][:100][::-1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=1,label='CFD')

for i in range(1,2):
    plt.plot(np.linspace(0,180,100),T[i][:100][::-1],'%s'%c[i],lw=2,label='%s'%mylabel[i])
    
plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.75), ncol=1, frameon=True, fancybox=False, shadow=False)
plt.xlabel('theta',fontsize=20)
plt.ylabel('$C_f$',fontsize=20)  
plt.savefig('./plot/cy_tau_%s.png'%kk, bbox_inches='tight',dpi=100)
plt.show()          

        
        

        
