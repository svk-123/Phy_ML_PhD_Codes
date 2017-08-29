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

#load
# duct - list
flist=['Re3500']

bD=[]
bR=[]
L=[]
T1= []
T2= []
T3= []
T4= []
T5= []
T6= []
x=[]
y=[]
z=[]


for ii in range(len(flist)):
   
    # for ref: data1 = [UUDi,bD,L,bR]
    with open('./tbnn_data/data_out/ml_data1_%s.pkl'%flist[ii], 'rb') as infile1:
        result1 = pickle.load(infile1)
    # ref: data2 = [T1m,T2m,T3m,T4m,T5m,T6m]
    with open('./tbnn_data/data_out/ml_data2_%s.pkl'%flist[ii], 'rb') as infile2:
        result2 = pickle.load(infile2)
    # ref: data2 = [x,y,z]
    with open('./tbnn_data/data_out/ml_data3_%s.pkl'%flist[ii], 'rb') as infile3:
        result3 = pickle.load(infile3)
    
    bD.extend(result1[1])
    bR.extend(result1[3])
    L.extend(result1[2])
    T1.extend(result2[0])
    T2.extend(result2[1])
    T3.extend(result2[2])
    T4.extend(result2[3])
    T5.extend(result2[4])
    T6.extend(result2[5])
    x.extend(result3[0])
    y.extend(result3[1])
    z.extend(result3[2])
    
bD=np.asarray(bD)
bR=np.asarray(bR)
L=np.asarray(L)
x=np.asarray(x)
y=np.asarray(y)
z=np.asarray(z)

T=[T1,T2,T3,T4,T5,T6]
T=np.asarray(T)

#load model
model_test = load_model('./model/final.hdf5') 
out=model_test.predict([L,T[0,:,:],T[1,:,:],T[2,:,:],T[3,:,:],T[4,:,:],T[5,:,:],])


# no of dot layers
l=6

# inverse scaler & reshape
out=np.asarray(out)
out=out.reshape(l,len(L))

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

for i in range(0,l):
    plot(z,y,bD[:,i],20,'%s'%(nbD[i]))
    #plot(z,y,bR[:,i],20,'%s'%(nbR[i]))   
    plot(z,y,out[i,:],20,'%s'%(nbp[i]))   
    #plot(z,y,t[:,i],20,'%s'%(nbp[i])) 

















