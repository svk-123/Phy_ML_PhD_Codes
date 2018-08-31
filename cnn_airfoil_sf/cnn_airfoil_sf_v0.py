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
path='./airfoil_data/'
data_file='foil_aoa_inout.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
print result[-1:]    


inp=result[0][:800]
out=result[2][:800]

inp=np.asarray(inp)
out=np.asarray(out)


xtr1=np.reshape(inp,(len(inp),288,216,1))  
ttr1=np.reshape(out,(len(out),288,216,1))  

del result
del inp
del out

# print dataset values
print('xtr shape:', xtr1.shape)
print('ttr shape:', ttr1.shape)

# Multilayer Perceptron
# create model
# construct model

# construct model
aa = Input([288,216,1])

# 2 3x3 convolutions followed by a max pool
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(aa)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

# 2 3x3 convolutions followed by a max pool
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

# 2 3x3 convolutions followed by a max pool
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

# 2 3x3 convolutions followed by a max pool
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

# 2 3x3 convolutions
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)


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


# 1 3x3 transpose convolution and concate conv4 on the depth dim
# ZeroPadding2D(top_pad, bottom_pad), (left_pad, right_pad)
up6 = concatenate([ZeroPadding2D(((0,0),(1,0)))(Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)), conv4], axis=3)

# 2 3x3 convolutions
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)



print ('up6',K.int_shape(up6))
print ('conv6',K.int_shape(conv6))


# 1 3x3 transpose convolution and concate conv3 on the depth dim
up7 = concatenate([ZeroPadding2D(((0,0),(0,0)))(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)), conv3], axis=3)

print ('up7',K.int_shape(up7))

# 2 3x3 convolutions
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

print ('conv7',K.int_shape(conv7))

# 1 3x3 transpose convolution and concate conv3 on the depth dim
up8 = concatenate([ZeroPadding2D(((0,0),(0,0)))(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)), conv2], axis=3)

print ('up8',K.int_shape(up8))

# 2 3x3 convolutions
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

print ('conv8',K.int_shape(conv8))

# 1 3x3 transpose convolution and concate conv3 on the depth dim
up9 = concatenate([ZeroPadding2D(((0,0),(0,0)))(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)), conv1], axis=3)

print ('up9',K.int_shape(up9))

# 2 3x3 convolutions
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

print ('conv9',K.int_shape(conv9))

# final 1x1 convolutions to get to the correct depth dim (3 for 2 xy vel and 1 for pressure)
conv10 = Conv2D(1, (1, 1), activation='linear')(conv9)

print ('conv10',K.int_shape(conv10))






#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[conv10])

#model = Model(inputs=[aa,t5], outputs=[y5])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=20, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=30, verbose=1, mode='auto')

filepath="./model_cnn/model_cnn_{epoch:02d}_{loss:.6f}_{val_loss:.6f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=25)

# Compile model
opt = Adam(lr=2.5e-5,decay=1e-12)

#scaler
#model.load_weights('./selected_model/naca4_cnn_ws/weight_model_af_cnn_100_0.003_0.004.hdf5')

model.compile(loss= 'mean_squared_error',optimizer= opt)

hist = model.fit([xtr1], [ttr1], validation_split=0.1,\
                 epochs=5000, batch_size=34,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=False)

#hist = model.fit([xtr0,xtr5], [ttr5], validation_split=0.3,\
#                 epochs=10000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1,shuffle=True)

#save model
model.save('./model_cnn/final_cnn.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))











