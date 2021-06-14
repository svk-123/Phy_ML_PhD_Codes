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
#
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

#B  = B_mm.transform(B)

bD=np.zeros((l,6))
bD[:,0]=bDtmp[:,0]
bD[:,1]=bDtmp[:,1]
bD[:,2]=bDtmp[:,2]
bD[:,3]=bDtmp[:,4]
bD[:,4]=bDtmp[:,5]
bD[:,5]=bDtmp[:,8]

dbD=np.zeros((l,6))
dbD[:,0]=bDtmp[:,0]-bRtmp[:,0]
dbD[:,1]=bDtmp[:,1]-bRtmp[:,1]
dbD[:,2]=bDtmp[:,2]-bRtmp[:,2]
dbD[:,3]=bDtmp[:,4]-bRtmp[:,4]
dbD[:,4]=bDtmp[:,5]-bRtmp[:,5]
dbD[:,5]=bDtmp[:,8]-bRtmp[:,8]

T=np.zeros((l,10,6))
T[:,:,0]=Ttmp[:,:,0]
T[:,:,1]=Ttmp[:,:,1]
T[:,:,2]=Ttmp[:,:,2]
T[:,:,3]=Ttmp[:,:,4]
T[:,:,4]=Ttmp[:,:,5]
T[:,:,5]=Ttmp[:,:,8]

#load model
model_test = load_model('./selected_model/4b/final_duct_L.hdf5') 
#out=model_test.predict([B,T[:,:,0],T[:,:,1],T[:,:,2],T[:,:,3],T[:,:,4],T[:,:,5]])
out=model_test.predict([B,T[:,0:4,0],T[:,0:4,1],T[:,0:4,2],T[:,0:4,3],T[:,0:4,4],T[:,0:4,5]])

# reshape
out=np.asarray(out)
out=out[:,:,0].transpose()
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

x=xyz[:,2]
y=xyz[:,1]
cor=[0,0,0,1,1,3]
for i in range(0,6):
    #plot(x,y,dbD[:,i],20,'%s'%(nbD[i]))
    k=i+cor[i]
    plot(x,y,bD[:,i],20,'%s'%(nbR[i]))   
    plot(x,y,out[:,i],20,'%s'%(nbp[i]))   
   # plot(z,y,sum(T[i,:,:].transpose()),20,'%s'%(nbp[i])) 
    #plot(x,y,bR[:,k],20,'%s'%(nbR[i]))   
    #plot(z,y,bDs[:],20,'%s'%(nbp[i]))   















