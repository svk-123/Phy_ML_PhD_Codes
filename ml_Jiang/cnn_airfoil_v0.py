#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

"""

import time
start_time = time.time()


# Python 3.5
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers import merge, Input, dot, add, concatenate
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import cPickle as pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

import os, shutil
folder = './model_cnn/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

# ref:[data,name]
path='./'
data_file='ml_input_output.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
my_inp=result[0]
my_out1=result[1]
my_out2=result[2]
my_out3=result[3]

my_out1=np.asarray(my_out1)
my_out2=np.asarray(my_out2)
my_out3=np.asarray(my_out3)

#CNN-ML
# ---------ML PART:-----------#
#shuffle data
N= len(my_inp)
I = np.arange(N)
np.random.shuffle(I)
n=20 # no. of training cases


## Training sets
xtr1 = my_inp[I][:n]
my_out=np.concatenate((my_out1[:,None],my_out2[:,None],my_out3[:,None]),axis=1)
my_out=my_out[:,:,0]
ttr1 = my_out[I][:n]
xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))         

# Multilayer Perceptron
# create model
# construct model

aa = Input([216,216,1])

# 2 3x3 convolutions followed by a max pool
conv1 = Conv2D(32, (4, 4), activation='relu', padding='same')(aa)
pool1 = MaxPooling2D(pool_size=(4, 4))(conv1)

# 2 3x3 convolutions followed by a max pool
conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(4, 4))(conv2)

conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

# flatten the 4D array (batch, height, width, depth) into 
# a 2D array (batch, n). Perform a fully connected layer
flat4 = Flatten()(pool3)

flat6 = Dense(50, activation='relu')(flat4)
flat6 = Dense(50, activation='relu')(flat6)
flat6 = Dense(50, activation='relu')(flat6)
# Dropout at 50% on this layer
#flat5_dropout = Dropout(0.1)(flat5)

# One more layer to a single value (this will be the predicted drag)
g = Dense(3, activation='linear')(flat6)

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[g])

#model = Model(inputs=[aa,t5], outputs=[y5])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min',verbose=1 ,patience=20, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='val_loss', min_delta=1.0e-8, patience=30, verbose=1, mode='auto')

filepath="./model_cnn/model_cnn_{epoch:02d}_{loss:.6f}_{val_loss:.6f}.hdf5"


chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=25)
# Compile model
opt = Adam(lr=2.5e-4,decay=1e-10)

#scaler
#model.load_weights('./selected_model/naca4_cnn_ws/weight_model_af_cnn_100_0.003_0.004.hdf5')

model.compile(loss= 'mean_squared_error',optimizer= opt)

hist = model.fit([xtr1], [ttr1], validation_split=0.2,\
                 epochs=100, batch_size=256,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=False)


#save model
model.save('./model_cnn/final_cnn.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))



data1=[hist.history]
with open('./model_cnn/hist.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)








