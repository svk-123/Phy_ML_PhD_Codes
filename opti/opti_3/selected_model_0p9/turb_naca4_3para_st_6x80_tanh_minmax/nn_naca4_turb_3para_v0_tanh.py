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
    
    data_file='../data_file/naca4_clcd_turb_st_3para.pkl'
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

inp_reno=np.asarray(inp_reno)
inp_aoa=np.asarray(inp_aoa)
inp_para=np.asarray(inp_para)/np.array([6,6,30])


n_out_cd=out_cd.copy()
n_out_cd[:]=0
n_out_cl=n_out_cd.copy()

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






# ---------ML PART:-----------#
#shuffle data
np.random.seed(123)
N= len(out_cm)
print N
I = np.arange(N)
np.random.shuffle(I)
n=18000

#normalize
inp_reno=inp_reno/100000.
inp_aoa=inp_aoa/14.0

my_inp=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
my_out=np.concatenate((n_out_cd[:,None],n_out_cl[:,None]),axis=1)



## Training sets
xtr0= my_inp[I][:n]
ttr1 = my_out[I][:n]

xts0= my_inp[I][n:]
tts1 = my_out[I][n:]

# Multilayer Perceptron
# create model
aa=Input(shape=(5,))
xx =Dense(80,  kernel_initializer='random_normal', activation='tanh')(aa)
xx =Dense(80, activation='tanh')(xx)
xx =Dense(80, activation='tanh')(xx)
xx =Dense(80, activation='tanh')(xx)
xx =Dense(80, activation='tanh')(xx)
xx =Dense(80, activation='tanh')(xx)
g =Dense(2, activation='linear')(xx)

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[g])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=100, min_lr=1.0e-7)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-7, patience=200, verbose=1, mode='auto')

filepath="./model/model_sf_{epoch:02d}_{loss:.8f}_{val_loss:.8f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=100)

# Compile model
opt = Adam(lr=5e-4,decay=1.0e-12)

model.compile(loss= 'mean_squared_error',optimizer= opt)

hist = model.fit([xtr0], [ttr1], validation_data=(xts0,tts1),\
                 epochs=5000, batch_size=64,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=False)

#save model
model.save('./model/final_sf.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))

data1=[hist.history]
with open('./model/hist.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)





















