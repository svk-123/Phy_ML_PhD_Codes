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
with open('../tbnn_v1//datafile/to_ml/ml_duct_Re3500_full.pkl', 'rb') as infile:
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

bD1=np.reshape(bD1,(len(bD1),1))
bD2=np.reshape(bD2,(len(bD2),1))
bD3=np.reshape(bD3,(len(bD3),1))
bD4=np.reshape(bD4,(len(bD4),1))
bD5=np.reshape(bD5,(len(bD5),1))
bD6=np.reshape(bD6,(len(bD6),1))


T=np.zeros((l,10,6))
T[:,:,0]=Ttmp[:,:,0]
T[:,:,1]=Ttmp[:,:,1]
T[:,:,2]=Ttmp[:,:,2]
T[:,:,3]=Ttmp[:,:,4]
T[:,:,4]=Ttmp[:,:,5]
T[:,:,5]=Ttmp[:,:,8]

T1=T[:,:,0]
T2=T[:,:,1]
T3=T[:,:,2]
T4=T[:,:,3]
T5=T[:,:,4]
T6=T[:,:,5]

scaler=False
#scaler=True
#scaler_start
if(scaler==True):
    name='B_bD1_to_bD6'
    with open('../tbnn_v1/scaler/%s.pkl'%name, 'rb') as infile:
        scaler= pickle.load(infile)
        
    B_mm=scaler[0]   
    B  = B_mm.transform(B)
     
    bD1_mm=scaler[1] 
    bD2_mm=scaler[2] 
    bD3_mm=scaler[3] 
    bD4_mm=scaler[4] 
    bD5_mm=scaler[5] 
    bD6_mm=scaler[6] 
    
    bD1  = bD1_mm.transform(bD1)
    bD2  = bD2_mm.transform(bD2)
    bD3  = bD3_mm.transform(bD3)
    bD4  = bD4_mm.transform(bD4)
    bD5  = bD5_mm.transform(bD5)
    bD6  = bD6_mm.transform(bD6)
#scaler_end


## Training sets
#xtr0 = L[I][:n]
xtr0 = B
xtr1 = T1
xtr2 = T2
xtr3 = T3
xtr4 = T4
xtr5 = T5
xtr6 = T6

ttr1 = bD1
ttr2 = bD2
ttr3 = bD3
ttr4 = bD4
ttr5 = bD5
ttr6 = bD6

ttr=np.concatenate((ttr1.reshape(len(ttr1),1),\
                    ttr2.reshape(len(ttr1),1),\
                    ttr3.reshape(len(ttr1),1),\
                    ttr4.reshape(len(ttr1),1),\
                    ttr5.reshape(len(ttr1),1),\
                    ttr6.reshape(len(ttr1),1)), axis=1)      

tmp=np.zeros((len(xtr0),2))
xtr0=np.concatenate((xtr0,tmp),axis=1)
xtr0=np.reshape(xtr0,(len(xtr0),7,7))
xtr0=np.reshape(xtr0,(len(xtr0),7,7,1))




#load model
model_test = load_model('./model_tbnn_cnn/final_tbnn_cnn.hdf5') 
#model_test = load_model('./selected_piml_cnn_model/model_piml_cnn_2399_0.058_0.056.hdf5') 
out=model_test.predict([xtr0,xtr1,xtr2,xtr3,xtr4,xtr5,xtr6])

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


#scaler_start
if(scaler==True):
    out[0,:,:]=bD1_mm.inverse_transform(out[0,:,:])
    out[1,:,:]=bD2_mm.inverse_transform(out[1,:,:])
    out[2,:,:]=bD3_mm.inverse_transform(out[2,:,:])
    out[3,:,:]=bD4_mm.inverse_transform(out[3,:,:])
    out[4,:,:]=bD5_mm.inverse_transform(out[4,:,:])
    out[5,:,:]=bD6_mm.inverse_transform(out[5,:,:])
#scaler_start

x=xyz[:,2]
y=xyz[:,1]
cor=[0,0,0,1,1,3]
for i in range(0,6):
    #plot(x,y,dbD[:,i],20,'%s'%(nbD[i]))
    k=i+cor[i]
    plot(x,y,bD[:,i],20,'%s'%(nbR[i]))   
    plot(x,y,out[i,:,0],20,'%s'%(nbp[i]))   
   # plot(z,y,sum(T[i,:,:].transpose()),20,'%s'%(nbp[i])) 
    #plot(x,y,bR[:,k],20,'%s'%(nbR[i]))   
    #plot(z,y,bDs[:],20,'%s'%(nbp[i]))   















