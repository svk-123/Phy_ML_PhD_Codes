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

# for ref: data=[L,T,bD,Coord,...]
#     ref:[x,tb,y,coord,k,ep,rans_bij,tkedns]
with open('./datafile/to_ml/ml_hill_Re10595_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
    
    
Ltmp.extend(result[0])
Ttmp.extend(result[1])
bDtmp.extend(result[2])
xyz.extend(result[3])
bR.extend(result[6])

    
bDtmp=np.asarray(bDtmp)
Ltmp=np.asarray(Ltmp)
Ttmp=np.asarray(Ttmp)
xyz=np.asarray(xyz)
bR=np.asarray(bR)

# reduce to 6 components
l=len(Ltmp)

L=Ltmp

bD=np.zeros((l,6))
bD[:,0]=bDtmp[:,0]
bD[:,1]=bDtmp[:,1]
bD[:,2]=bDtmp[:,2]
bD[:,3]=bDtmp[:,4]
bD[:,4]=bDtmp[:,5]
bD[:,5]=bDtmp[:,8]

T=np.zeros((l,10,6))
T[:,:,0]=Ttmp[:,:,0]
T[:,:,1]=Ttmp[:,:,1]
T[:,:,2]=Ttmp[:,:,2]
T[:,:,3]=Ttmp[:,:,4]
T[:,:,4]=Ttmp[:,:,5]
T[:,:,5]=Ttmp[:,:,8]


#stdscaler
#L  = L_ss.transform(L)


#load model
model_test = load_model('./model/final.hdf5') 
out=model_test.predict([L,T[:,:,0],T[:,:,1],T[:,:,2],T[:,:,3],T[:,:,4],T[:,:,5]])
    

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

x=xyz[:,0]
y=xyz[:,1]
cor=[0,0,0,1,1,3]
for i in range(0,6):
    plot(x,y,bD[:,i],20,'%s'%(nbD[i]))
    k=i+cor[i]
    #plot(x,y,bR[:,k],20,'%s'%(nbR[i]))   
    plot(x,y,out[:,i],20,'%s'%(nbp[i]))   
   # plot(z,y,sum(T[i,:,:].transpose()),20,'%s'%(nbp[i])) 
    #plot(z,y,bR[:,i],20,'%s'%(nbR[i]))   
    #plot(z,y,bDs[:],20,'%s'%(nbp[i]))   















