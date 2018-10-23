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
with open('./data_file/foil_aoa_nn_p16_ph_1_tr_21.pkl', 'rb') as infile:
    result = pickle.load(infile)
inp_x=result[0]   
inp_y=result[1]   
inp_para=result[2]   
inp_aoa=result[3]   
out_p=result[4]   

inp_x=np.asarray(inp_x)
inp_y=np.asarray(inp_y)
inp_para=np.asarray(inp_para)
inp_aoa=np.asarray(inp_aoa)
out_p=np.asarray(out_p)


# ---------ML PART:-----------#
#shuffle data
N= len(inp_x)
I = np.arange(N)
np.random.shuffle(I)
n=700000

#normalize
inp_aoa=inp_aoa/12.0

my_inp=np.concatenate((inp_x[:,None],inp_y[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
my_out=out_p


## Training sets
xtr0= my_inp[I][:n]
ttr1 = my_out[I][:n]


del result
del inp_x
del inp_y
del inp_para
del inp_aoa
del out_p





# Multilayer Perceptron
# create model
aa=Input(shape=(19,))
xx =Dense(100,  kernel_initializer='random_normal', activation='relu')(aa)
xx =Dense(100, activation='relu')(xx)
xx =Dense(100, activation='relu')(xx)
xx =Dense(100, activation='relu')(xx)
xx =Dense(100, activation='relu')(xx)
xx =Dense(100, activation='relu')(xx)
xx =Dense(100, activation='relu')(xx)
xx =Dense(100, activation='relu')(xx)
g =Dense(1, activation='linear')(xx)

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[g])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=100, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=200, verbose=1, mode='auto')

filepath="./model/model_sf_{epoch:02d}_{loss:.6f}_{val_loss:.6f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=50)

# Compile model
opt = Adam(lr=2.5e-5,decay=1.0e-12)

model.compile(loss= 'mean_squared_error',optimizer= opt)

hist = model.fit([xtr0], [ttr1], validation_split=0.2,\
                 epochs=5000, batch_size=10000,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=False)

#save model
model.save('./model/final_sf.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))

























