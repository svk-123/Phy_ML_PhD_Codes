#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:31:27 2017

@author: vino
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 19:35:45 2017

@author: vino
"""

# imports
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import  cm

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.layers import merge, Input, Dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import cPickle as pickle

#time 
import time
start_time = time.time()

#imports
from read_rans import read_rans
from write_R import write_R

x,z,y,k,L,T1m,T2m,T3m,T4m,T5m,T6m=read_rans()

#ml-model-pred
#load model
model_test = load_model('../model/model_9999_0.170_0.177.hdf5') 
bpred=model_test.predict([L,T1m,T2m,T3m,T4m,T5m,T6m])

# inverse scaler & reshape
bpred=np.asarray(bpred)
bpred=bpred.reshape(6,len(L)) 


write_R(bpred[0,:],bpred[1,:],bpred[2,:],bpred[3,:],bpred[4,:],bpred[5,:])



