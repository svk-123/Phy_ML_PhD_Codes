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
import cPickle as pickle
import pandas

from scipy import interpolate
from numpy import linalg as LA
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#load para
with open('./n0012/n0012_para.pkl', 'rb') as infile:
    para = pickle.load(infile)
para=np.asarray(para[0])

#load model
model_test=load_model('./n0012/model_sf_400_0.00000737_0.00000795.hdf5') 
    
def get_force(reno,aoa):

    reno=1000    

    tmp=np.loadtxt('./n0012/n0012.dat',skiprows=1)
    inp_x=tmp[:,0]
    inp_y=tmp[:,1]
    
    inp_reno=np.repeat(reno/1000., len(inp_x))
    inp_aoa=np.repeat(aoa/20., len(inp_x))   
    inp_para=np.repeat(para[:,None],len(inp_x),axis=1).transpose()   
    
    val_inp=np.concatenate((inp_x[:,None],inp_y[:,None],inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
    out=model_test.predict([val_inp]) 
         

    a0=find_nearest(tmp[:,0],0)
    co=tmp.copy()
    xu=co[:a0+1,0]
    yu=co[:a0+1,1]
    xl=co[a0:,0]
    yl=co[a0:,1]
    
    #pressure
    pu2=out[0:a0+1,0]
    pl2=out[a0:,0]
       
   
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
        if(dy[j] <=0):
            dF.append(0.5*cp[j]*abs(dy[j]))
        else:                
            dF.append(-0.5*cp[j]*abs(dy[j]))       
            
    cl=sum(lF)/(0.5)
    cd=sum(dF)/(0.5)
    
    return (cl,cd)

#test
tmp=[]
aoalist=np.linspace(-15,15,50)    
for k in range(len(aoalist)):
    tmp.append(get_force(1000,aoalist[k]))
tmp=np.asarray(tmp)
x=aoalist    
plt.plot(x,tmp[:,0],label='cl')
plt.plot(x,tmp[:,1],label='cd')
plt.xlabel('aoa',fontsize=16)
plt.ylabel('cl-cd',fontsize=16)
plt.legend()
plt.show()    