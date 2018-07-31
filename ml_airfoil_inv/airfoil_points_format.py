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
from scipy import interpolate


#co-fomatting
path='./airfoil_1600_1aoa_1re/naca'

indir=path
outdir='./airfoil_1600_1aoa_1re/naca131'
fname = [f for f in listdir(indir) if isfile(join(indir, f))]

nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])
   
xx=np.loadtxt('./airfoil_1600_1aoa_1re/n0012.dat')      

xxu=xx[:66,0]
xxl=xx[66:,0]

    
for i in range(len(fname)):
    
    print i
    coord=np.loadtxt(path+'/%s'%fname[i],skiprows=1)
        
    l=len(coord)
    ind=((l-1)/2)
    
    up_x=coord[:ind+1,0]
    up_y=coord[:ind+1,1]
        
    lr_x=coord[ind:,0]
    lr_y=coord[ind:,1]    
        
    up_x[0]=1
    up_x[-1:]=0
        
    lr_x[0]=0    
    lr_x[-1:]=1
    
    fu = interpolate.interp1d(up_x, up_y)
    u_yy = fu(xxu)
        
    fl = interpolate.interp1d(lr_x, lr_y)
    l_yy = fl(xxl)      
    
    fp= open(outdir+"/%s"%fname[i],"w+")
    fp.write('%s\n'%nname[i])    
    
    for j in range(len(xxu)):
        fp.write("%f %f\n"%(xxu[j],u_yy[j])) 
        
    for j in range(len(xxl)):
        fp.write("%f %f\n"%(xxl[j],l_yy[j])) 
        
    fp.close()
    
    
     
    
    
    
    
    
    
    
    
    
    
##coordinate-check
#indir='./coord_seligFmt_formatted'
#fname = [f for f in listdir(indir) if isfile(join(indir, f))]
#coord=[]
#for i in range(len(fname)):
#    coord.append(np.loadtxt(indir+'/%s'%fname[i],skiprows=1))
    
    
    
    
    
    
    
    
    
    
    
    

