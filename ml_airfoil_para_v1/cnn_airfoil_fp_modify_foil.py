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
from matplotlib import cm
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
from numpy import linalg as LA
import os, shutil
from scipy.interpolate import interp1d
 
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
plt.rc('font', family='serif')

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

# ref:[data,name]

inp=[]
out=[]


path='./data_file/'
data_file='foil_uiuc.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp=result[0]
out=result[1]
xx=result[2]
name=result[3]

inp=np.asarray(inp)
my_out=np.asarray(out)

xtr1=inp
ttr1=my_out 

xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))  
model=load_model('./selected_model/case_1_p8_tanh_uiuc_aug_gan0p5_naca4_v2/model_cnn/final_cnn.hdf5') 

del inp
del result

# with a Sequential model

get_out_1c= K.function([model.layers[0].input], [model.layers[15].output])
get_foil= K.function([model.layers[16].input],  [model.layers[19].output])

latent = get_out_1c([xtr1])[0]


# morphed airfoils
cnt=0    
for k in range(100):
        
        I= np.random.randint(0,100)
        l=latent[I].reshape(1,8)
        
        noise=np.random.normal(0,0.5,8).reshape(1,8)
        #original
        out1=get_foil([l])[0]
        out1=out1*0.2
        #original
        out2=get_foil([l+noise])[0]
        out2=out2*0.2
        
        if(1 > 0):
            #plot
            plt.figure()
            plt.plot(xx[:100],out1[0,:],'r',label='original')
            plt.plot(xx[:100],out2[0,:],'g',label='morphed')
            #plt.axis('off')
            plt.ylim([-0.2,0.2])
            plt.legend(fontsize=12)
            plt.savefig("plot/true_%d.png" % k)
            
            plt.show()
            plt.close()     
            cnt=cnt+1

