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

#read excel file
dataframe=pandas.read_excel('airfoil_v1.xlsx', sheet_name=0, header=0, skiprows=None)
dataset = dataframe.values
mydata=np.asarray(dataset)

#format the data
tmpcol=mydata[:,6]
mylist=[]
for i in range(len(tmpcol)):
    mylist.append(map(float,tmpcol[i].split(',')))
mylist=np.asarray(mylist)    
mydata=np.concatenate((mydata, mylist), axis=1)
mydata=np.delete(mydata,[0,6],axis=1)

tmplist=[]
for i in range(len(mydata)):
    if(mydata[i,4]==5):
        tmplist.append(i)
        
mydata=np.delete(mydata,tmplist,axis=0)        

#train - validation split
my_inp=mydata[:-10,[0,1,2,3,6]]
my_out=mydata[:-10,5]

val_inp=mydata[-10:,[0,1,2,3,6]]
val_out=mydata[-10:,5]

#normalize
my_inp[:,2]=my_inp[:,2]/2.
val_inp[:,2]=val_inp[:,2]/2.

my_inp[:,3]=my_inp[:,3]/50000.
val_inp[:,3]=val_inp[:,3]/50000.

my_out=my_out/100.
val_out=val_out/100.


#test-val
model=load_model('./model/final.hdf5')
out=model.predict([val_inp])

#re-normalize
val_out=val_out*100
out=out*100

#plot
plt.figure(figsize=(8, 5), dpi=100)
plt0, =plt.plot(val_out,'og',label='true')
plt1, =plt.plot(out,'or',label='nn')  
plt.legend()
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.01,1.4)    
plt.savefig('ml_airfoil_0', format='png', dpi=100)
plt.show() 






















