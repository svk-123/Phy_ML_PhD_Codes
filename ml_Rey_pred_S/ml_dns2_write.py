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

from scipy import interpolate

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
    with open('../tbnn_data/data_out/ml_data1_%s.pkl'%flist[ii], 'rb') as infile1:
        result1 = pickle.load(infile1)
    # ref: data2 = [T1m,T2m,T3m,T4m,T5m,T6m]
    with open('../tbnn_data/data_out/ml_data2_%s.pkl'%flist[ii], 'rb') as infile2:
        result2 = pickle.load(infile2)
    # ref: data2 = [x,y,z]
    with open('../tbnn_data/data_out/ml_data3_%s.pkl'%flist[ii], 'rb') as infile3:
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
model_test = load_model('../model/model_9999_0.170_0.177.hdf5') 
out=model_test.predict([L,T[0,:,:],T[1,:,:],T[2,:,:],T[3,:,:],T[4,:,:],T[5,:,:]])

# inverse scaler & reshape
out=np.asarray(out)
out=out.reshape(6,len(L))

with open('../rans_data/duct_rans_data1_%s.pkl'%flist[0], 'rb') as infile1:
    result4 = pickle.load(infile1)
k=result4[6]

a11=out[0,:]*2*k
a12=out[1,:]*2*k
a13=out[2,:]*2*k
a22=out[3,:]*2*k
a23=out[4,:]*2*k
a33=out[5,:]*2*k

t11=a11+(2./3.)*k
t12=a12
t13=a13
t22=a22+(2./3.)*k
t23=a23
t33=a33+(2./3.)*k


from ml_Rey_write import write_R_ml
write_R_ml(t11,t12,t13,t22,t23,t33)

























