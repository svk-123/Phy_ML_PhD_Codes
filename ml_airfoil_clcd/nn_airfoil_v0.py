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

#Re, d1, d2, d3, alp, cl, cd
data2=[]
for i in range(len(name)):
    data2.extend(data1[i])

data2=np.asarray(data2)    

my_inp=data2[:,0:5]
my_inp[:,0]=my_inp[:,0]*10
my_inp[:,3]=my_inp[:,3]/2.
my_out=data2[:,5]

# ---------ML PART:-----------#
#shuffle data
N= len(my_inp)
I = np.arange(N)
np.random.shuffle(I)
n=2000

## Training sets
xtr0= my_inp[I][:n]
ttr1 = my_out[I][:n]

# Multilayer Perceptron
# create model


aa=Input(shape=(5,))
xx =Dense(10,  kernel_initializer='random_normal', activation='relu')(aa)
xx =Dense(30, activation='relu')(xx)
xx =Dense(30, activation='relu')(xx)
xx =Dense(30, activation='relu')(xx)
xx =Dense(30, activation='relu')(xx)
g =Dense(1, activation='linear')(xx)

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[g])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, mode='min',verbose=1 ,patience=1000, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=1000, verbose=1, mode='auto')

filepath="./model/model_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=1000)

# Compile model
opt = Adam(lr=2.5e-5,decay=1.0e-12)

model.compile(loss= 'mean_squared_error',optimizer= opt)

hist = model.fit([xtr0], [ttr1], validation_split=0.1,\
                 epochs=10000, batch_size=16,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=False)

#save model
model.save('./model/final.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))


data1=[hist.history]
with open('./model_cnn/hist.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)






















