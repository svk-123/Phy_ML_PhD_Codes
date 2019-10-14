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

import os, shutil



#load data
inp_reno=[]
inp_aoa=[]
inp_para=[]

out_cm=[]
out_cd=[]
out_cl=[]

for ii in [1]:
    
    data_file='./data_file/naca4_clcd_turb_st_3para.pkl'
    with open(data_file, 'rb') as infile:
        result = pickle.load(infile)

	out_cm.extend(result[0])   
	out_cd.extend(result[1])
    	out_cl.extend(result[2])

    	inp_reno.extend(result[3])
	inp_aoa.extend(result[4])
	inp_para.extend(result[5])
    
#    data_file='./data_file/naca4_clcd_turb_ust_3para.pkl'
#    with open(data_file, 'rb') as infile:
#        result = pickle.load(infile)
#
#	out_cm.extend(result[0])   
#	out_cd.extend(result[1])
#    	out_cl.extend(result[2])
#
#    	inp_reno.extend(result[3])
#	inp_aoa.extend(result[4])
#	inp_para.extend(result[5])

out_cm=np.asarray(out_cm)
out_cd=np.asarray(out_cd)
out_cl=np.asarray(out_cl)

n_out_cd=out_cd.copy()
n_out_cd[:]=0
n_out_cl=n_out_cd.copy().copy()

"""###
scaler min-max
out_cl: max= 1.45, min=-0.35
out_cd: max= 0.25, min=0.01 
###"""

mini = -1.0
maxi =  1.0

for i in range(len(out_cd)):
    
    n_out_cd[i] = ((maxi-mini)/(0.25-0.01))*(out_cd[i]-0.25) + maxi
    n_out_cl[i] = ((maxi-mini)/(1.45+0.35))*(out_cl[i]-1.45) + maxi

inp_reno=np.asarray(inp_reno)
inp_aoa=np.asarray(inp_aoa)
inp_para=np.asarray(inp_para)/np.array([6,6,30])


#normalize
inp_reno=inp_reno/100000.
inp_aoa=inp_aoa/14.0

my_inp=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
my_out=np.concatenate((out_cd[:,None],out_cl[:,None]),axis=1)
n_my_out=np.concatenate((n_out_cd[:,None],n_out_cl[:,None]),axis=1)

model_test=load_model('./selected_model/turb_naca4_3para_st_6x80_tanh_minmax/model/final_sf.hdf5') 
n_out=model_test.predict([my_inp]) 
out=n_out.copy()
out[:,:]=0

for i in range(len(out)):
    
    out[i,0] = ((0.25-0.01)/(maxi-mini))*(n_out[i,0]-maxi)  + 0.25
    out[i,1] = ((1.45+0.35)/(maxi-mini))*(n_out[i,1]-maxi)  + 1.45


plt.figure(figsize=(6,5),dpi=100)
plt.plot([-0.5,1.5],[-0.5,1.5],'k',lw=3)
plt.plot(my_out[:,1],out[:,1],'o',markersize=4)
#plt.xlim([-0.05,0.25])
#plt.legend(fontsize=20)
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)  
#plt.savefig('./plot/ts_%s_%s.png'%(k,name[k]), bbox_inches='tight',dpi=100)
plt.show()


plt.figure(figsize=(6,5),dpi=100)
plt.plot([0.01,0.06],[0.01,0.06],'k',lw=3)
plt.plot(my_out[:,0],out[:,0],'o',markersize=1)
#plt.xlim([-0.05,0.25])
#plt.legend(fontsize=20)
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)  
#plt.savefig('./plot/ts_%s_%s.png'%(k,name[k]), bbox_inches='tight',dpi=100)
plt.show()
















