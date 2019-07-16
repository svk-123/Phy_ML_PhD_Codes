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
import matplotlib.tri as tri
from matplotlib import cm
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

from scipy import interpolate
from numpy import linalg as LA
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
plt.rc('font', family='serif')

#load data
inp_x=[]
inp_y=[]
inp_reno=[]
inp_aoa=[]
inp_para=[]

out_p=[]
out_u=[]
out_v=[]

#load data
#with open('./data_file/ph_1_test/foil_aoa_nn_p16_ph_1_ts_1.pkl', 'rb') as infile:
with open('./data_file/foil_aoa_nn_nacan_lam_trts_1.pkl', 'rb') as infile:
    result = pickle.load(infile,encoding='bytes')

inp_x.extend(result[0])   
inp_y.extend(result[1])
inp_para.extend(result[2])
inp_reno.extend(result[3])
inp_aoa.extend(result[4])

out_p.extend(result[5])
out_u.extend(result[6])
out_v.extend(result[7])

co=result[8]
fxy=result[9]
name=result[9]

inp_x=np.asarray(inp_x)
inp_y=np.asarray(inp_y)
inp_reno=np.asarray(inp_reno)
inp_aoa=np.asarray(inp_aoa)
inp_para=np.asarray(inp_para)

out_p=np.asarray(out_p)
out_u=np.asarray(out_u)
out_v=np.asarray(out_v)
