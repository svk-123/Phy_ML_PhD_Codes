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
import  pickle
import pandas

import os, shutil


#load data
inp_x=[]
inp_y=[]
inp_reno=[]
inp_aoa=[]
inp_para=[]
inp_t=[]

out_p=[]
out_u=[]
out_v=[]


for ii in range(1):
    data_file='./data_file/cy_un_lam_tr_1_1112.pkl'

    with open(data_file, 'rb') as infile:
        result = pickle.load(infile)

    inp_x.extend(result[0])   
    inp_y.extend(result[1])
    inp_reno.extend(result[2])
    inp_t.extend(result[3])
    
    out_p.extend(result[4])
    out_u.extend(result[5])
    out_v.extend(result[6])

inp_x=np.asarray(inp_x)
inp_y=np.asarray(inp_y)
inp_reno=np.asarray(inp_reno)
inp_t=np.asarray(inp_t)
out_p=np.asarray(out_p)
out_u=np.asarray(out_u)
out_v=np.asarray(out_v)

