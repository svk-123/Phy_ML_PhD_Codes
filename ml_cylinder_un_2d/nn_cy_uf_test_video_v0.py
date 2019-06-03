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
import pickle
import pandas

from scipy import interpolate
from numpy import linalg as LA
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

#load data
inp_x=[]
inp_y=[]
inp_reno=[]
inp_t=[]

out_p=[]
out_u=[]
out_v=[]

#load data
with open('./data_file/cy_un_lam_ts_1.pkl', 'rb') as infile:
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

co=np.zeros((100,2))
theta=np.linspace(0,360,100)
for i in range(len(theta)):
    co[i,0]= 0.5*np.cos(np.radians(theta[i]))
    co[i,1]= 0.5*np.sin(np.radians(theta[i]))    
    
#open pdf file

#plot
def con_plot(i):
    
    fig = plt.figure(figsize=(8, 4),dpi=100)
        
    ax1 = fig.add_subplot(1,2,1)
    cp1 = ax1.tricontourf(val_inp[:,0],val_inp[:,1],val_out[:,1],20,cmap=cm.jet)
    ax1.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax1.set_title('CFD')
    #ax1.set_xlabel('None')
    #ax1.set_ylabel('None')
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_xlim([-3,8])
    ax1.set_ylim([-3,3])
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #cbar1=plt.colorbar(cp1, cax=cax, orientation='vertical');
    #cbar1.ax.tick_params(labelsize=10)
    #plt.subplots_adjust( wspace=0.2,hspace=0.0)
    ax1.set_aspect(1)
    
    ax2 = fig.add_subplot(1,2,2)
    cp2 = ax2.tricontourf(val_inp[:,0],val_inp[:,1],out[:,1],20,cmap=cm.jet)
    ax2.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax2.set_title('NN')
    #ax2.set_xlabel('X',fontsize=16)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim([-3,8])
    ax2.set_ylim([-3,3])
    #divider = make_axes_locatable(ax2)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #cbar2=plt.colorbar(cp2, cax=cax, orientation='vertical');
    #cbar2.ax.tick_params(labelsize=10)
    ax2.set_aspect(1)
        
      
    fig.suptitle('T = %0.4f'%inp_t[i][0])
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)       
    plt.savefig('./plot_1/%02d.png'%i,format='png',dpi=100)
    plt.close()


#model 
model_test=load_model('./selected_model/case_1_8x500/model_sf_130_0.00000190_0.00000267.hdf5') 
    
np.random.seed(1234534)
mylist=range(24)

for i in mylist:
    
    print (i)
    
    #normalize
    inp_reno[i]=inp_reno[i]/1000.
    
    val_inp=np.concatenate((inp_x[i][:,None],inp_y[i][:,None],inp_reno[i][:,None],inp_t[i][:,None]),axis=1)
    val_out=np.concatenate((out_p[i][:,None],out_u[i][:,None],out_v[i][:,None]),axis=1)

    #load_model
    out=model_test.predict([val_inp]) 
        
    con_plot(i)
    #LinearNDinterpolator
    pD=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()
               
#    #for -p
#    print ('interpolation-1...')      
#    f1p=interpolate.LinearNDInterpolator(pD,val_out[:,0])
#        
#    pu1=np.zeros(len(xu))
#    for j in range(len(xu)):
#        pu1[j]=f1p(xu[j],yu[j])
#    pl1=np.zeros(len(xl))
#    for j in range(len(xl)):
#        pl1[j]=f1p(xl[j],yl[j])
#        
#    print ('interpolation-2...')     
#    f2p=interpolate.LinearNDInterpolator(pD,out[:,0])
#      
#    pu2=np.zeros(len(xu))
#    for j in range(len(xu)):
#        pu2[j]=f2p(xu[j],yu[j])
#    pl2=np.zeros(len(xl))
#    for j in range(len(xl)):
#       pl2[j]=f2p(xl[j],yl[j])
    



    



