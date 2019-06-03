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
    
    data_file='./data_file/naca4_clcd_turb_st_8para.pkl'
    with open(data_file, 'rb') as infile:
        result = pickle.load(infile)

	out_cm.extend(result[0])   
	out_cd.extend(result[1])
    	out_cl.extend(result[2])

    	inp_reno.extend(result[3])
	inp_aoa.extend(result[4])
	inp_para.extend(result[5])
    
    data_file='./data_file/gen_clcd_turb_st_8para.pkl'
    with open(data_file, 'rb') as infile:
        result = pickle.load(infile)

	out_cm.extend(result[0])   
	out_cd.extend(result[1])
    	out_cl.extend(result[2])

    	inp_reno.extend(result[3])
	inp_aoa.extend(result[4])
	inp_para.extend(result[5])

out_cm=np.asarray(out_cm)
out_cd=np.asarray(out_cd)/0.33
out_cl=np.asarray(out_cl)/2.04

inp_reno=np.asarray(inp_reno)
inp_aoa=np.asarray(inp_aoa)
inp_para=np.asarray(inp_para)

#normalize
inp_reno=inp_reno/100000.
inp_aoa=inp_aoa/14.0

my_inp=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
my_out=np.concatenate((out_cd[:,None],out_cl[:,None]),axis=1)

model_test=load_model('./selected_model/turb_gen_8para_6x50/model_sf_1600_0.00003714_0.00004874.hdf5') 
out=model_test.predict([my_inp]) 

my_out[:,0]=my_out[:,0] * 0.33
my_out[:,1]=my_out[:,1] * 2.04

out[:,0]=out[:,0] * 0.33
out[:,1]=out[:,1] * 2.04

plt.figure(figsize=(6,5),dpi=100)
plt.plot([-0.0,0.35],[-0.0,0.35],'k',lw=3)
plt.plot(my_out[:,0],out[:,0],'og',markersize=3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('True $C_d$',fontsize=20)
plt.ylabel('Predicted $C_d$',fontsize=20)  
plt.savefig('./plot/gen_cd.png', bbox_inches='tight',dpi=100)
plt.show()



















