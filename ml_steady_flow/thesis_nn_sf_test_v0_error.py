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
import cPickle as pickle
import pandas

from scipy import interpolate
from numpy import linalg as LA
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

#load data
xtmp=[]
ytmp=[]
reytmp=[]
utmp=[]
vtmp=[]
ptmp=[]

flist=['Re400']
for ii in range(len(flist)):
    #x,y,Re,u,v
    with open('./data/cavity_%s.pkl'%flist[ii], 'rb') as infile:
        result = pickle.load(infile)
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    reytmp.extend(result[2])
    utmp.extend(result[3])
    vtmp.extend(result[4])
    ptmp.extend(result[5])    
    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
reytmp=np.asarray(reytmp)
utmp=np.asarray(utmp)
vtmp=np.asarray(vtmp)    
ptmp=np.asarray(ptmp) 

#normalize
reytmp=reytmp/1000.
val_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
val_out=np.concatenate((utmp[:,None],vtmp[:,None],ptmp[:,None]),axis=1)    

#load_model
model_test=load_model('./thesis_selected_model/case_1_8x100_tanh/model/final_sf.hdf5') 
out=model_test.predict([val_inp])    


train_l2_p=[]
train_l2_u=[]
train_l2_v=[]

tmp=val_out[:,0]-out[:,0]
train_l2_p.append( (LA.norm(tmp)/LA.norm(out[:,0]))*100 )

tmp=val_out[:,1]-out[:,1]
train_l2_u.append( (LA.norm(tmp)/LA.norm(out[:,1]))*100 )

tmp=val_out[:,2]-out[:,2]
train_l2_v.append( (LA.norm(tmp)/LA.norm(out[:,2]))*100 )



