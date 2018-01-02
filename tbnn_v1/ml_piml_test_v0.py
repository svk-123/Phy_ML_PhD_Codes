#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

@author: vinoth
"""
import time
start_time = time.time()

# Python 3.5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import  cm

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.layers import merge, Input, Dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import cPickle as pickle
import seaborn as sns

import os,sys
scriptpath = "/home/vino/miniconda2/mypy"
sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)

Ltmp=[]
Ttmp=[]
bDtmp=[]
xyz=[]
bR=[]
Btmp=[]
bRtmp=[]
wdtmp=[]

# ref:[x,tb,y,coord,k,ep,rans_bij,tkedns,I,B,wd]
with open('./datafile/to_ml/ml_duct_Re3500_full.pkl', 'rb') as infile:
    result = pickle.load(infile)

Ltmp.extend(result[0])
Ttmp.extend(result[1])
bDtmp.extend(result[2])
xyz.extend(result[3])
bR.extend(result[6])
Btmp.extend(result[9])
bRtmp.extend(result[6])
wdtmp.extend(result[10])
    
bDtmp=np.asarray(bDtmp)
Ltmp=np.asarray(Ltmp)
Ttmp=np.asarray(Ttmp)
xyz=np.asarray(xyz)
bR=np.asarray(bR)
Btmp=np.asarray(Btmp)
bRtmp=np.asarray(bRtmp)
wdtmp=np.asarray(wdtmp)

# reduce to 6 components
# reduce to 6 components
l=len(Ltmp)

L=Ltmp
B=Btmp

name='B_bD1_to_bD6'
with open('./scaler/%s.pkl'%name, 'rb') as infile:
    scaler= pickle.load(infile)
    
B_mm=scaler[0] 
#B  = B_mm.transform(B)

   
bD1_mm=scaler[1] 
bD2_mm=scaler[2] 
bD3_mm=scaler[3] 
bD4_mm=scaler[4] 
bD5_mm=scaler[5] 
bD6_mm=scaler[6] 

bD=np.zeros((l,6))

bD[:,0]=bDtmp[:,0]
bD[:,1]=bDtmp[:,1]
bD[:,2]=bDtmp[:,2]
bD[:,3]=bDtmp[:,4]
bD[:,4]=bDtmp[:,5]
bD[:,5]=bDtmp[:,8]

bD1=bD[:,0]
bD2=bD[:,1]
bD3=bD[:,2]
bD4=bD[:,3]
bD5=bD[:,4]
bD6=bD[:,5]

'''
bD1=np.reshape(bD1,(len(bD1),1))
bD2=np.reshape(bD2,(len(bD2),1))
bD3=np.reshape(bD3,(len(bD3),1))
bD4=np.reshape(bD4,(len(bD4),1))
bD5=np.reshape(bD5,(len(bD5),1))
bD6=np.reshape(bD6,(len(bD6),1))

bD1  = bD1_mm.transform(bD1)
bD2  = bD2_mm.transform(bD2)
bD3  = bD3_mm.transform(bD3)
bD4  = bD4_mm.transform(bD4)
bD5  = bD5_mm.transform(bD5)
bD6  = bD6_mm.transform(bD6)
'''

#load model
model_test = load_model('./model_piml/2_final_piml.hdf5') 
out=model_test.predict([B])

# reshape
out=np.asarray(out)
#out=out[:,:,0] #if single

#plot
def plot(x,y,z,nc,name):
    fig=plt.figure(figsize=(6, 5), dpi=100)
    ax=fig.add_subplot(111)
    #cp = ax.tricontourf(x, y, z,np.linspace(-0.3,0.3,30),extend='both')
    cp = ax.tricontourf(x, y, z,30,extend='both')
    #cp.set_clim(-0.2,0.2)
    #plt.xlim([-1, 0])
    #plt.ylim([-1, 0])
     
    cbar=plt.colorbar(cp)
    plt.title(name)
    plt.xlabel('Z ')
    plt.ylabel('Y ')
    #plt.savefig(name +'.png', format='png', dpi=100)
    plt.show()

nbD=['uu-bD','uv-bD','uw-bD','vv-bD','vw-bD','ww-bD']
nbp=['uu-pred','uv-pred','uw-pred','vv-pred','vw-pred','ww-pred']
nbR=['uu-bR','uv-bR','uw-bR','vv-bR','vw-bR','ww-bR']
#nbD=['uu-bD','uv-bD','uw-bD','vu-bD','vv-bD','vw-bD','wu-bD','wv-bD','ww-bD']

#import scipy
#out=scipy.ndimage.filters.gaussian_filter(out,0.1,mode='nearest')


#bD2=bD2_mm.inverse_transform(bD2)
#out=bD2_mm.inverse_transform(out)

x=xyz[:,2]
y=xyz[:,1]
cor=[0,0,0,1,1,3]
for i in range(0,1):
    #plot(x,y,dbD[:,i],20,'%s'%(nbD[i]))
    k=i+cor[i]
    plot(x,y,bD2,20,'%s'%(nbR[i]))   
    plot(x,y,out[:,0],20,'%s'%(nbp[i]))   
   # plot(z,y,sum(T[i,:,:].transpose()),20,'%s'%(nbp[i])) 
    #plot(x,y,bR[:,k],20,'%s'%(nbR[i]))   
    #plot(z,y,bDs[:],20,'%s'%(nbp[i]))   















