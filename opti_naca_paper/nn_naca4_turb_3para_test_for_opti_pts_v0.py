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



   

inp_reno=np.asarray([20000])
inp_aoa=np.asarray([6])
inp_para=np.asarray([4,5,12])/np.array([6.,6.,30.])
#normalize
inp_reno=inp_reno/100000.
inp_aoa=inp_aoa/14.0

my_inp=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,None]),axis=0).transpose()

nname='relu-6x80'
model_test=load_model('./selected_model_0p9/turb_naca4_3para_st_6x80_relu/model/final_sf.hdf5') 

#cd, cl
out=model_test.predict([my_inp]) 

out[0][0]=out[0][0]*0.25
out[0][1]=out[0][1]*0.9

print (out)
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
