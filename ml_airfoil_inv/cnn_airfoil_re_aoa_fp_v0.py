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

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten,UpSampling2D
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
data_file='data_cp_re_aoa_fp.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp_up=result[0]
inp_lr=result[1]
out=result[2]
reno=result[3]
aoa=result[4]
xx=result[5]
name=result[6]

inp_up=np.asarray(inp_up)
inp_lr=np.asarray(inp_lr)
out=np.asarray(out)
out=out/0.16

xtr1=np.concatenate((inp_up[:,:,:,None],inp_lr[:,:,:,None]),axis=3) 
ttr1=out 

del result
del inp_up
del inp_lr
del aoa
del xx
del reno
del out

# print dataset values
print('xtr shape:', xtr1.shape)
print('ttr shape:', ttr1.shape)

# Multilayer Perceptron
# create model
# construct model
aa = Input([216,216,2])

# 2 3x3 convolutions followed by a max pool
conv1 = Conv2D(32, (4, 4), activation='relu', padding='same')(aa)
conv1 = Conv2D(32, (4, 4), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(3, 3))(conv1)

conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(3, 3))(conv2)

conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(3, 3))(conv3)

conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(256, (2, 2), activation='relu', padding='same')(pool4)
pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

# flatten the 4D array (batch, height, width, depth) into 
# a 2D array (batch, n). Perform a fully connected layer
flat5 = Flatten()(pool5)

flat5 = Dense(100, activation='relu')(flat5)
flat5 = Dense(100, activation='relu')(flat5)
flat5 = Dense(100, activation='relu')(flat5)

g = Dense(70, activation='linear')(flat5)



#print
print ('aa',K.int_shape(aa))
print ('conv1',K.int_shape(conv1))
print ('pool1',K.int_shape(pool1))
print ('conv2',K.int_shape(conv2))
print ('pool2',K.int_shape(pool2))
print ('conv3',K.int_shape(conv3))
print ('pool3',K.int_shape(pool3))
print ('conv4',K.int_shape(conv4))
print ('pool4',K.int_shape(pool4))
print ('conv5',K.int_shape(conv5))
print ('pool5',K.int_shape(pool5))
print ('flat5',K.int_shape(flat5))

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa],outputs=[g])

#model = Model(inputs=[aa,t5], outputs=[y5])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=20, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-6, patience=30, verbose=1, mode='auto')

filepath="./model_cnn/model_enc_cnn_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"
filepath_weight="./model_cnn/weight_model_enc_cnn_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=50)

chkpt_weight= ModelCheckpoint(filepath_weight, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=True, mode='auto', period=50)
# Compile model
opt = Adam(lr=2.5e-4,decay=1e-10)

#scaler
#model.load_weights('./selected_model/naca4_cnn_ws/weight_model_af_cnn_100_0.003_0.004.hdf5')

model.compile(loss= 'mean_squared_error',optimizer= opt)
#model.compile(loss= 'binary_crossentropy',optimizer= opt,metrics=['accuracy'])


hist = model.fit([xtr1], [ttr1], validation_split=0.1,\
                 epochs=5000, batch_size=32,callbacks=[reduce_lr,e_stop,chkpt,chkpt_weight],verbose=1,shuffle=False)

#hist = model.fit([xtr0,xtr5], [ttr5], validation_split=0.3,\
#                 epochs=10000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1,shuffle=True)

#save model
model.save('./model_cnn/final_enc_cnn.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))











