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
import pickle
import pandas

import os, shutil
from numpy import linalg as LA



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
    
out_cm=np.asarray(out_cm)
out_cd=np.asarray(out_cd)/0.25
out_cl=np.asarray(out_cl)/0.9

inp_reno=np.asarray(inp_reno)
inp_aoa=np.asarray(inp_aoa)
inp_para=np.asarray(inp_para)
#inp_para=inp_para/np.array([6,6,30])

#normalize
#inp_reno=inp_reno/100000.
#inp_aoa=inp_aoa/14.0
#
#my_inp=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
#my_out=np.concatenate((out_cd[:,None],out_cl[:,None]),axis=1)
#
#my_inp1=[]
#my_out1=[]
#for i in range(len(my_inp)):
#    if (my_inp[i,0]==50000 and my_inp[i,2]==2 and my_inp[i,3]==4 and my_inp[i,4]==12 ):
#        my_inp1.append(my_inp[i,:])
#        my_out1.append(my_out[i,:])
#        
#my_inp1=np.asarray(my_inp1)
#my_out1=np.asarray(my_out1)
#
#my_inp1[:,0]=my_inp1[:,0]/100000.
#my_inp1[:,1]=my_inp1[:,1]/14.
#my_inp1[:,2:5]=my_inp1[:,2:5]/np.array([6,6,30])
#
#nname='relu-6x80'
#model_test=load_model('./selected_model_0p9/turb_naca4_3para_st_6x80_relu/model/final_sf.hdf5') 
#out=model_test.predict([my_inp1]) 
#
#my_inp1[:,1]=my_inp1[:,1]*14.
#
#out[:,0]=out[:,0]*0.25
#out[:,1]=out[:,1]*0.9
#
#my_out1[:,0]=my_out1[:,0]*0.25
#my_out1[:,1]=my_out1[:,1]*0.9

inp_reno=np.repeat(50000,8)
inp_aoa=np.array([0,2,4,6,8,10,12,14])

tmp=[6,6,06]
inp_para=[]
for i in range(8):
    inp_para.append(tmp)
    
inp_para=np.asarray(inp_para)/np.array([6.,6.,30.])    
inp_reno=inp_reno/100000.
inp_aoa=inp_aoa/14.
    
my_inp=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
model_test=load_model('./selected_model_0p9/turb_naca4_3para_st_6x80_tanh/model/final_sf.hdf5') 
out=model_test.predict([my_inp]) 

out[:,0]=out[:,0]*0.25
out[:,1]=out[:,1]*0.9



for i in range(1):
    
    plt.figure(figsize=(6, 6), dpi=100)
        
    plt0, =plt.plot(my_inp[:,1]*14,out[:,1],'o',mfc='b',mew=1.5,mec='b',ms=10,label='CFD-$C_l$')
    plt0, =plt.plot(my_inp[:,1]*14,out[:,0],'s',mfc='b',mew=1.5,mec='b',ms=10,label='CFD-$C_d$') 
                
    plt0, =plt.plot(my_inp[:,1]*14,out[:,1],'r',lw=3,marker='o',mfc='None',mew=1.5,mec='r',ms=10,label='MLP-relu-$C_l$')
    plt0, =plt.plot(my_inp[:,1]*14,out[:,0],'r',lw=3,marker='s',mfc='None',mew=1.5,mec='r',ms=10,label='MLP-relu-$C_d$')     

    plt0, =plt.plot(my_inp[:,1]*14,out[:,1],'--g',lw=3,marker='o',mfc='None',mew=1.5,mec='g',ms=10,label='MLP-tanh-$C_l$')
    plt0, =plt.plot(my_inp[:,1]*14,out[:,0],'--g',lw=3,marker='s',mfc='None',mew=1.5,mec='g',ms=10,label='MLP-tanh-$C_d$') 
      
    plt.xlabel('AoA',fontsize=20)
    plt.ylabel('$C_l$,$C_d$' ,fontsize=20)
    #plt.title('%s-AoA-%s-p'%(flist[ii],AoA[jj]),fontsize=16)
    plt.subplots_adjust(top = 0.95, bottom = 0.2, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
    plt.figtext(0.45, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
    plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(-0.0,0.75), ncol=1, frameon=False, fancybox=False, shadow=False)
    #plt.xlim(-0.3,7)
    #plt.ylim(-0.01,0.5) 
    #plt.xticks([0,4,8,12])
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.savefig('./plot/clcd_6606.tiff', format='tiff',bbox_inches='tight', dpi=300)
    plt.show()    
    plt.close() 