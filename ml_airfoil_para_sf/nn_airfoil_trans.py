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
with open('./data_file/foil_aoa_nn_test_ts_p16.pkl', 'rb') as infile:
    result = pickle.load(infile)
inp_x=result[0]   
inp_y=result[1]   
inp_para=result[2]   
inp_aoa=result[3]   
out_p=result[4]   
out_u=result[5] 
out_v=result[6] 

co=result[7]
co=co[700:]

name=result[8]

inp_x=np.asarray(inp_x)
inp_y=np.asarray(inp_y)
inp_para=np.asarray(inp_para)
inp_aoa=np.asarray(inp_aoa)
out_p=np.asarray(out_p)
out_u=np.asarray(out_u)
out_v=np.asarray(out_v)

#plot
def con_plot(xp,yp,zp,nc,i,pname):

    plt.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
    plt.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
    #v= np.linspace(0, 0.05, 15, endpoint=True)
    #cp = plt.tricontourf(xp,yp,zp,v,cmap=cm.jet,extend='both')
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    plt.colorbar(cp)
    #plt.title('%s  '%flist[ii]+name)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    plt.savefig('./plot_c_ts/%s_%s_%s_aoa_%s.eps'%(i,name[i][0],pname,val_inp[2,0]), format='eps',bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()


for i in range(1):
    
    inp_aoa[i]=inp_aoa[i]/12.0
    val_inp=np.concatenate((inp_x[i][:,None],inp_y[i][:,None],inp_aoa[i][:,None],inp_para[i][:,:]),axis=1)
    val_out=np.concatenate((out_p[i][:,None],out_u[i][:,None],out_v[i][:,None]),axis=1)
         
    con_plot(val_inp[:,0],val_inp[:,1],val_out[:,0],20,i,'p-cfd')


