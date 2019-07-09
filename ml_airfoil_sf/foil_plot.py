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
xtmp=[]
ytmp=[]
reytmp=[]
aoatmp=[]
utmp=[]
vtmp=[]
ptmp=[]

flist=['Re100']
AoA=range(1)
foil='n0012'

for ii in range(len(flist)):
    for jj in range(len(AoA)):
        #x,y,Re,u,v
        with open('./data/%s_%s_AoA_%s.pkl'%(foil,flist[ii],AoA[jj]), 'rb') as infile:
            result = pickle.load(infile)
        xtmp.extend(result[0])
        ytmp.extend(result[1])
        reytmp.extend(result[2])
        aoatmp.extend(result[3])
        utmp.extend(result[4])
        vtmp.extend(result[5])
        ptmp.extend(result[6])   
    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
reytmp=np.asarray(reytmp)
aoatmp=np.asarray(aoatmp)
utmp=np.asarray(utmp)
vtmp=np.asarray(vtmp)
ptmp=np.asarray(ptmp) 



