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
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from skimage import io, viewer,util 

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
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)



#co-fomatting
'''
path='./coord_seligFmt_original'

indir=path
outdir='./coord_seligFmt_formatted'
fname = [f for f in listdir(indir) if isfile(join(indir, f))]
nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])
    
#load coord
for i in range(len(fname)):
#for i in range(10):
    with open(path+'/%s'%fname[i], 'r') as infile:
        data0=infile.readlines()

    fp= open(outdir+"/%s"%fname[i],"w+")
    fp.write('%s\n'%nname[i])    
    for j in range(1,len(data0)):
        fp.write("%s"%(data0[j])) 

    fp.close()
    
'''    
    
#coordinate-check
indir='./coord_seligFmt_formatted'
fname = [f for f in listdir(indir) if isfile(join(indir, f))]
coord=[]
for i in range(len(fname)):
    coord.append(np.loadtxt(indir+'/%s'%fname[i],skiprows=1))
    
    
    
    
    
    
    
    
    
    
    
    

