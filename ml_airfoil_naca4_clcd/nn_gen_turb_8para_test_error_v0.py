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

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
#plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rc('font', family='serif')

#load data
inp_reno=[]
inp_aoa=[]
inp_para=[]

out_cm=[]
out_cd=[]
out_cl=[]

for ii in [1]:
    
    data_file='./data_file_v1/gen_cnn_clcd_turb_8para_v1.pkl'
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

nname='tanh-6x80'
model_test=load_model('./selected_model/case_gen_naca4_80x6/model/final_sf.hdf5')  
out=model_test.predict([my_inp]) 

np.random.seed(1234)

I = np.arange(len(my_inp))
np.random.shuffle(I)
n=70000

my_out_train= my_out[I][:n] 
my_out_val = my_out[I][n:] 

out_train= out[I][:n]
out_val = out[I][n:]

'''tmp1=abs(my_out_train[:,0]-out_train[:,0])
cd_train=(LA.norm(tmp1)/LA.norm(my_out_train[:,0]) )*100 
tmp2=abs(my_out_val[:,0]-out_val[:,0])
cd_val=(LA.norm(tmp2)/LA.norm(my_out_val[:,0]) )*100 

tmp3=abs(my_out_train[:,1]-out_train[:,1])
cl_train=(LA.norm(tmp3)/LA.norm(my_out_train[:,1]) )*100 
tmp4=abs(my_out_val[:,1]-out_val[:,1])
cl_val=(LA.norm(tmp4)/LA.norm(my_out_val[:,1]) )*100 

print('cd_train', cd_train)
print('cd_val', cd_val)
print('cl_train', cl_train)
print('cl_val', cl_val)'''


tmp1=abs(my_out_train[:,0]-out_train[:,0])
I1=np.argwhere(tmp1 < 0.05)


tmp2=abs(my_out_train[:,1]-out_train[:,1])
I2=np.argwhere(tmp1 < 0.2)




l1=1
l2=1
plt.figure(figsize=(6,5),dpi=100)
plt.plot([-0.0,0.33],[-0.0,0.33],'k',lw=2)
plt.plot(my_out_train[I1,0]*0.33,out_train[I1,0]*0.33,'g',marker='+',mew=1.5, mfc='None',markevery=l1,lw=0,ms=8,label='Train')
plt.plot(my_out_val[:,0]*0.33,out_val[:,0]*0.33,'r',marker='+',mew=1.5, mfc='None',markevery=l2,lw=0,ms=6,label='Test')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16,frameon=False, shadow=False, fancybox=False)
plt.xlabel('True $C_D$',fontsize=20)
plt.figtext(0.50, -0.07, '(a)', wrap=True, horizontalalignment='center', fontsize=24) 
plt.ylabel('Predicted $C_D$',fontsize=20)  
plt.savefig('./plot/naca_cd-%s.png'%(nname),format='png' ,bbox_inches='tight',dpi=300)
plt.show()


l1=1
l2=1
plt.figure(figsize=(6,5),dpi=100)
plt.plot([-0.6,2.1],[-0.6,2.1],'k',lw=2)
plt.plot(my_out_train[I1,1]*2.04,out_train[I1,1]*2.04,'g',marker='+',mew=1.5, mfc='None',markevery=l1,lw=0,ms=8,label='Train')
plt.plot(my_out_val[:,1]*2.04,out_val[:,1]*2.04,'r',marker='+',mew=1.5, mfc='None',markevery=l2,lw=0,ms=6,label='Test')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16,frameon=False, shadow=False, fancybox=False)
plt.figtext(0.50, -0.07, '(b)', wrap=True, horizontalalignment='center', fontsize=24) 
plt.xlabel('True $C_L$',fontsize=20)
plt.ylabel('Predicted $C_L$',fontsize=20)  
plt.savefig('./plot/naca_cl-%s.png'%(nname),format='png' ,bbox_inches='tight',dpi=300)
plt.show()
#
#
#
#
#
#
#
#
#
#
