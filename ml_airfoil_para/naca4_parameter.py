#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
reads naca4 foils and gives 3 parameters,i.e its digits

@author: vinoth
"""

import time
start_time = time.time()

# Python 3.5
import numpy as np
import matplotlib.pyplot as plt
import pandas
from os import listdir
from os.path import isfile, join

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers import merge, Input, dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import  pickle
import pandas

import os, shutil

indir="./coord_naca4_opti"
fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()

name=[]
for i in range(len(fname)):        
    name.append(fname[i].split(".",1)[0]) 
                

d1=[]
d2=[]
d3=[]
#split space from name
for i in range(len(name)):
    tmp0=name[i].split()
    tmp1=list(tmp0[0])
 
    if (len(tmp1)==2):
        d1.append(0)
        d2.append(0)
        d3.append(float(tmp1[1]))
        
    elif (len(tmp1)==3):
        d1.append(0)
        d2.append(0)
        d3.append(float(tmp1[1]+tmp1[2]))
        
    elif (len(tmp1)==5):
        d1.append(float(tmp1[1]))
        d2.append(float(tmp1[2]))
        d3.append(float(tmp1[3]+tmp1[4]))    
        
    elif (len(tmp1)==8):
        d1.append(float(tmp1[4]))
        d2.append(float(tmp1[5]))
        d3.append(float(tmp1[6]+tmp1[7]))   

d1=np.asarray(d1)
d2=np.asarray(d2)
d3=np.asarray(d3)

para=np.concatenate((d1[:,None],d2[:,None],d3[:,None]),axis=1)

scaler= np.asarray([6,6,30])

para_scaled=para.copy()

para_scaled[:,0]=para_scaled[:,0]/scaler[0]
para_scaled[:,1]=para_scaled[:,1]/scaler[1]
para_scaled[:,2]=para_scaled[:,2]/scaler[2]

# ref:[x,y,z,ux,uy,uz,k,ep,nut]
info=['para_scaled, scaler, para, name, info- naca 400+ airfoils name used ']

data1 = [para_scaled,scaler,para,name,info ]

with open('./naca4_digit_para_opti_foil.pkl', 'wb') as outfile1:
    pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)








