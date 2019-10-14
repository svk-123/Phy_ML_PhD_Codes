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
folder = './model/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)


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
#out_cd=np.asarray(out_cd)/0.25
#out_cl=np.asarray(out_cl)/0.9

out_cd=np.asarray(out_cd)
out_cl=np.asarray(out_cl)

n_out_cd=out_cd.copy()
n_out_cd[:]=0
n_out_cl=n_out_cd.copy()
nn_out_cd=n_out_cd.copy()
nn_out_cl=n_out_cd.copy()

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


#m_cd=np.mean(n_out_cd)
#s_cd=np.std(n_out_cd)
#
#m_cl=np.mean(n_out_cl)
#s_cl=np.std(n_out_cl)

"""
m-mean, s-std dev
#m_cd = -0.56, s_cd =0.39
#m_cl = -0.15, s_cl=0.31
"""
m_cd = -0.56
s_cd = 0.39
m_cl = -0.15
s_cl = 0.31

for i in range(len(out_cd)):
    
    nn_out_cd[i] = (n_out_cd[i]-m_cd)/s_cd
    nn_out_cl[i] = (n_out_cl[i]-m_cl)/s_cl




plt.figure(figsize=(6,5),dpi=100)
plt.hist(out_cl, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.ylabel('Number of Samples',fontsize=20)
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.show()

plt.figure(figsize=(6,5),dpi=100)
plt.hist(nn_out_cl, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.ylabel('Number of Samples',fontsize=20)
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.show()


plt.figure(figsize=(6,5),dpi=100)
plt.hist(out_cd, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.ylabel('Number of Samples',fontsize=20)
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.show()

plt.figure(figsize=(6,5),dpi=100)
plt.hist(nn_out_cd, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.ylabel('Number of Samples',fontsize=20)
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.show()