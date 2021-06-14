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



fname_1=['bl_ml']

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

Re=10000

#ws-review sample loc
#fname_2=np.asarray([ ['bl_%s_0'%Re,'bl_100_0_nodp_nodv_x8','bl_%s_0_sample_p8'%Re,'bl_%s_0_sample_p10'%Re] ])   
 
#ws-review noise
#fname_2=np.asarray([ ['bl_%s_0'%Re,'bl_100_0_nodp_nodv_x8','bl_%s_0_sample_p8'%Re,'bl_%s_0_sample_p10'%Re] ]) 

#BC-1,2,3 (wos)
#fname_2=np.asarray([ ['bl_%s_0'%Re,'bl_%s_0_dp_dv'%Re,'bl_%s_0_nodp_nodv'%Re,'bl_%s_0_wos_4s_puv'%Re] ]) 

#ws-S1,S2
#fname_2=np.asarray([ ['bl_%s_0'%Re,'bl_%s_0_nodp_nodv_x8_away'%Re,'bl_%s_0_nodp_nodv_x8'%Re] ]) 

#ws-res
fname_2=np.asarray([ ['bl_%s_0'%Re,'bl_%s_0_nodp_nodv_x8'%Re] ]) 

myxx=np.loadtxt('./bl_ml/xx.dat')

'''
Re100-629
Re1000-452
Re5000-301
Re10000-237
Re20000-188
Re40000-153
'''



tmp=[]
foil=[]
for i in range(len(fname_1)):
    for j in range(len(fname_2[i])):
        tmp.append(fname_2[i][j])
        foil.append(fname_2[i][j].split('_')[0])
tmp=np.asarray(tmp)    
foil=np.asarray(foil)

#mylabel=['CFD','BC-1','BC-2','BC-3']
#mylabel=['CFD','S-1','S-2']
mylabel=['CFD','PINN']
#mylabel=['CFD','PINN-sample @ 50%','PINN-sample @ 20%','PINN-sample @ 0%' ]
#mylabel=['CFD','PINN','PINN-noise 5%','PINN-noise 10%' ]


#mu=5/float(Re)
nu=1/float(Re)
xx=myxx
rex=xx/nu
cf=0.664/np.sqrt(rex)

T=[]
for jj in range(1):
    
        #load shearstress 
        tu1=[]
        tu2=[]

        with open('./bl_ml/%s/237/wallShearStress'%(tmp[jj]), 'r') as infile:
            
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
        T.append(abs(tu1)/0.5)  
      
   
c=['g','g','b','r','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown']          
plt.figure(figsize=(6,5),dpi=100)
#plt.plot(myxx,T[0][:],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=3,label='CFD')
plt.plot(myxx,cf,'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=3,label='Blasius')
#plt.plot(xx,cf)
for i in range(1):
    plt.plot(myxx,T[i][:],'%s'%c[i],lw=2,label='%s'%mylabel[i])
    
plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(0.2,0.75), ncol=1, frameon=False, fancybox=False, shadow=False)
plt.xlabel('X',fontsize=20)
plt.ylabel('$C_f$',fontsize=20)  
#plt.ylim([-0.05, max(T[0])*1.1])
plt.ylim([-0.005,0.1])
plt.savefig('./plot/%s.tiff'%fname_2[0][1], format='tiff', bbox_inches='tight',dpi=300)
plt.show()          
        
        
        

        
