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



fname_1=['foil_ml']

fname_1=np.asarray(fname_1)
fname_1.sort()


# read data from below dir...
path='.'
#path='/home/vino/ml_test/from_nscc_26_dec_2018/foam_run/case_naca_lam'
indir = path

#np.random.seed(1234)
#np.random.shuffle(fname)

#fname_2=[]
#for i in range(len(fname_1)):
#    dir2=indir + '/%s'%fname_1[i]
#    tmp=[f for f in listdir(dir2) if isdir(join(dir2, f))]
#    fname_2.append(tmp)

Re=100  
fname_2=np.asarray([ ['naca4412_000100_20','naca4412_000100_20_pinn_nodp_nodv_ar_30x2'] ])    
#fname_2.sort()






tmp=[]
foil=[]
for i in range(len(fname_1)):
    for j in range(len(fname_2[i])):
        tmp.append(fname_2[i][j])
        foil.append(fname_2[i][j].split('_')[0])
tmp=np.asarray(tmp)    
foil=np.asarray(foil)

mylabel=['CFD','BC-1','BC-2','BC-3']

mylabel=['CFD','Samp-1','Samp-2','BC-3']
#mylabel=['CFD','PINN']

mu=1/float(Re)
T=[]
for jj in range(2):
    
        #load shearstress 
        tu1=[]
        tu2=[]

        with open('./foil_ml/%s/327/wallShearStress'%(tmp[jj]), 'r') as infile:
            
            data0=infile.readlines()
            ln=0
            for line in data0:
                if 'foil' in line:
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
        T.append(mu*abs(tu1)/0.5)  
      


xx=np.loadtxt('naca4412_x.dat')   

c=['g','g','b','r','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown']          
plt.figure(figsize=(6,5),dpi=100)
plt.plot(xx[:138],T[0][:138],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=3,label='CFD')

for i in range(1,2):
    plt.plot(xx[:138],T[i][:138],'%s'%c[i],lw=2,label='%s'%mylabel[i])
    
plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(0.5,0.75), ncol=1, frameon=False, fancybox=False, shadow=False)
plt.xlabel('X',fontsize=20)
plt.ylabel('$C_f$',fontsize=20)  
plt.ylim([-0.0, 0.01])
#plt.xlim([0.0, 7])
plt.savefig('./plot/%s.png'%fname_2[0][1], bbox_inches='tight',dpi=100)
plt.show()          
        
        
        

        
