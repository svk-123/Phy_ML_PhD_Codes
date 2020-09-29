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
from matplotlib import cm
from os import listdir
from os.path import isfile, join

import keras
from keras import backend as K
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
import cPickle as pickle

from scipy import interpolate
from numpy import linalg as LA
import matplotlib


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


path='./data_file/'
data_file='param_216_16.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)
para=result1[0][0]
name=result1[1]

para=np.asarray(para)
name=np.asarray(name)

idx=np.argwhere(name=='naca0024')[0][0]
my_para=para[1155:1156,:]
tmp_para=para[1155,:]

#get airfoil coord from parameters
model_para=load_model('./selected_model/p16/model_cnn_2950_0.000013_0.000176.hdf5')  
get_out_1c= K.function([model_para.layers[12].input],[model_para.layers[15].output])
# requires input shape (1,16)
c1 = get_out_1c([my_para])[0][0,:]
xx=tmp=np.loadtxt('./data_file/xx.txt')
co=np.zeros((69,2))
co[0:35,0]=xx[::-1]
co[0:35,1]=c1[:35][::-1]
co[35:69,0]=xx[1:]
co[35:69,1]=c1[36:]

#plt.figure(figsize=(6,5),dpi=100)
#plt.plot(co[:,0],co[:,1],'r',label='true')
#plt.show()


#predict flow and calcyulate cl
model_flow=load_model('./selected_model/case_9_naca_lam_1/model_sf_65_0.00000317_0.00000323.hdf5') 

inp_x=co[:,0]
inp_y=co[:,1]

reno=600
aoa=12.0
    
inp_reno=np.repeat(reno/2000., len(inp_x))
inp_aoa=np.repeat(aoa/14., len(inp_x))   
inp_para=np.repeat(tmp_para[:,None],len(inp_x),axis=1).transpose()   
#reqires shape others-(70,1) & para-(70,16) & val_inp (70,20)    
val_inp=np.concatenate((inp_x[:,None],inp_y[:,None],inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
out=model_flow.predict([val_inp]) 
         
#a0=find_nearest(co[:,0],0)
a0=34
xu=co[:a0+1,0]
yu=co[:a0+1,1]
xl=co[a0:,0]
yl=co[a0:,1]
    
#pressure
pu2=out[:a0+1,0]
pl2=out[a0:,0]

plt.figure(figsize=(6,5),dpi=100)
plt.plot(co[:,0],out[:,0],'r',label='true')
plt.show()
          
#cl calculation        
xc=[]
yc=[]
dx=[]
dy=[]
    
pc=[]
for j in range(len(xu)-1): 
    pc.append((pu2[j]+pu2[j+1])/2.0)
for j in range(len(xl)-1): 
    pc.append((pl2[j]+pl2[j+1])/2.0)
        
for j in range(len(co)-1):
    xc.append((co[j,0] + co[j+1,0])/2.0)
    yc.append((co[j,1] + co[j+1,1])/2.0)    
        
    dx.append((co[j+1,0] - co[j,0]))
    dy.append((co[j+1,1] - co[j,1]))    
    
cp=[]    
for j in range(len(xc)):
    cp.append(2*pc[j])
     
lF=[]
for j in range(len(xc)):
    if(dx[j] <=0):
        lF.append(-0.5*cp[j]*abs(dx[j]))
    else:                
        lF.append(0.5*cp[j]*abs(dx[j]))
dF=[]            
for j in range(len(xc)):
    dF.append(-0.5*cp[j]*dy[j])
            
cl=sum(lF)*np.cos(np.radians(aoa))/(0.5)
cd=sum(dF)/(0.5)
    



