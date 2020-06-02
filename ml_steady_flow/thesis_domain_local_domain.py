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
plt.rc('font', family='serif')

#load data
xtmp=[]
ytmp=[]
reytmp=[]
utmp=[]
vtmp=[]
ptmp=[]

flist=['Re1000']
for ii in range(len(flist)):
    #x,y,Re,u,v
    with open('./data/cavity_%s_part_y.pkl'%flist[ii], 'rb') as infile:
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


x=[0,1,1,0,0]
y=[0,0,1,1,0]

x1=[0,1,1,0,0]
y1=[0.45,0.45,0.55,0.55,0.45]
#domain part
plt.figure(figsize=(5, 5), dpi=100)
plt0, =plt.plot(x,y,'k',linewidth=3)
plt.tricontourf(xtmp,ytmp,utmp,zorder=2)
plt0, =plt.plot(x1,y1,'r',linewidth=3,zorder=3)

###text-1
#plt.text(2.5, -0.3, "Wall: u=0", horizontalalignment='center', verticalalignment='center')
#plt.text(2.5, 3.3, "Outlet: p=0", horizontalalignment='center', verticalalignment='center')
#plt.text(-0.3, 1.5, "Inlet: u-specified", horizontalalignment='center', verticalalignment='center',rotation=90)

##text-2
#plt.text(2.5, -0.3, "Wall: u=0,dp=0", horizontalalignment='center', verticalalignment='center')
#plt.text(2.5, 3.3, "Outlet: p=0,du=0", horizontalalignment='center', verticalalignment='center')
#plt.text(-0.3, 1.5, "Inlet: u-specified, dp=0", horizontalalignment='center', verticalalignment='center',rotation=90)

##text-2
#plt.text(2.5, -0.3, "Wall: u=0,p-specified", horizontalalignment='center', verticalalignment='center')
#plt.text(2.5, 3.3, "Outlet: p=0,u-specified", horizontalalignment='center', verticalalignment='center')
#plt.text(-0.3, 1.5, "Inlet: u, p-specified", horizontalalignment='center', verticalalignment='center',rotation=90)

#plt.legend(fontsize=20)
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
#plt.title('%s-u'%(flist[ii]),fontsiuze=16)
plt.legend(loc='upper center', bbox_to_anchor=(1.5, 1), ncol=1, fancybox=False, shadow=False,fontsize=16)
plt.xlim(0,1)
plt.ylim(0,1)    
plt.savefig('mesh1.png', format='png',bbox_inches='tight', dpi=200)
plt.show()



