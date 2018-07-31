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

from numpy import linalg as LA
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

       
indir="./naca4/polar_train"
fname = [f for f in listdir(indir) if isfile(join(indir, f))]

#read polar
#dataframe=pandas.read_csv(indir+'/%s'%fname[0], header=0, skiprows=None)
#dataset = dataframe.values
#mydata=np.asarray(dataset)

name=[]   
rey_no=[]
data1=[]  
for i in range(len(fname)):       
    with open(indir+'/%s'%fname[i],'r') as myfile:
        data0=myfile.readlines()
    
        if "Calculated polar for:" in data0[2]:
                name.append(data0[2].split("NACA",1)[1]) 
    
        if "Re =" in data0[7]:
                tmp=data0[7].split("Re =",1)[1]
                rey_no.append(tmp.split("e",1)[0])
    
    #load alpha cl cd
    tmp_data=np.loadtxt(indir+'/%s'%fname[i],skiprows=11)
    data1.append(tmp_data[:,0:3])

d1=[]
d2=[]
d3=[]
#split space from name
for i in range(len(name)):
    tmp0=name[i].split()
    tmp1=list(tmp0[0])
    d1.append(float(tmp1[0]))
    d2.append(float(tmp1[1]))
    d3.append(float(tmp1[2]+tmp1[3]))

d1=np.asarray(d1)
d2=np.asarray(d2)
d3=np.asarray(d3)

#rey_no
for i in range(len(rey_no)):
    tmp0=rey_no[i].split()
    rey_no[i]=float(tmp0[0])

for i in range(len(name)):
    tmp0=np.full(len(data1[i]),rey_no[i])
    tmp1=np.full(len(data1[i]),d1[i])
    tmp2=np.full(len(data1[i]),d2[i])
    tmp3=np.full(len(data1[i]),d3[i])
    
    data1[i]=np.concatenate((tmp0[:,None],tmp1[:,None],tmp2[:,None],tmp3[:,None],data1[i]),axis=1)
    
    
my_error=[]    
#load_model
model_test=load_model('./selected_model/naca4_nn/final.hdf5') 

#spread plot
plt.figure(figsize=(6, 5), dpi=100) 
plt0, =plt.plot([-0.05,1.6],[-0.05,1.6],'k',lw=3) 

for i in range(len(data1)):
#for i in range(1):    
    #Re, d1, d2, d3, alp, cl, cd
    data2=data1[i]
    
    val_inp=data2[:,0:5]
    val_inp[:,0]=val_inp[:,0]*10
    val_inp[:,3]=val_inp[:,3]/2.
    val_out=data2[:,5]
    
    
    #test-val
    out=model_test.predict([val_inp])

    
    #plot
    '''plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(val_inp[:,4],val_out,'-og',linewidth=3,label='true')
    plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=3,label='NN')  
    plt.legend(fontsize=20)
    plt.xlabel('AoA',fontsize=20)
    plt.ylabel('$C_l$',fontsize=20)
    #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4) 

    plt.savefig('./train/NACA%sRe=%se6'%(name[i],rey_no[i]), format='png',bbox_inches='tight', dpi=100)
    plt.show() '''
    
    
    #spread
    #plt0, =plt.plot(val_out,out,'og') 
  
#plt.legend(fontsize=20)
'''plt.xlabel('True $C_l$',fontsize=20)
plt.ylabel('Predicted $C_l$',fontsize=20)
#plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.01,1.4) 

plt.savefig('val1', format='png',bbox_inches='tight', dpi=100)    
plt.show()


path='./selected_model/naca4_nn'
data_file='/hist.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
history=result[0]
#hist
plt.figure(figsize=(6,5),dpi=100)
plt.plot(range(len(history['loss'])),history['loss'],'r',lw=3,label='training error')
plt.plot(range(len(history['val_loss'])),history['val_loss'],'b',lw=3,label='validation error')
plt.legend(fontsize=20)
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
#plt.xlim([-10,5000])
#plt.ylim([-0.2,0.2])    
plt.savefig('convergence.png')
plt.show()'''


endtime=time.time()






    
#Error
'''tmp1=abs(out-val_out[:,None])
tmp2=LA.norm(tmp1)/LA.norm(val_out)
tmp3=(tmp2)*100
my_error.append(tmp3)
print tmp3'''
    




















