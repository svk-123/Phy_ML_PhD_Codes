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
import  pickle
import pandas

from scipy import interpolate
from numpy import linalg as LA
import matplotlib
import tensorflow as tf

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 



# Load Data
# Load Data
#load data

xtmp=[]
ttmp=[]
tbtmp=[]
titmp=[]
eptmp=[]
    
for ii in range(1,2):
    #x,y,Re,u,v
    tmp=np.loadtxt('./1d_train1/T%d'%ii,delimiter=',')

    xtmp.extend(tmp[:,0])
    titmp.extend(tmp[:,1])
    tbtmp.extend(tmp[:,2])
    ttmp.extend(tmp[:,3])
    eptmp.extend(tmp[:,4])
    

xtmp=np.asarray(xtmp)
titmp=np.asarray(titmp)
tbtmp=np.asarray(tbtmp)
ttmp=np.asarray(ttmp)
eptmp=np.asarray(eptmp)
       
x  = xtmp[:,None] # NT x 1
ti = titmp[:,None] # NT x 1

tb = tbtmp[:,None] # NT x 1
t  = ttmp[:,None] # NT x 1
ep = eptmp[:,None] # NT x 1
#session-run
graph = tf.get_default_graph() 
#load model
with tf.Session() as sess:
   
    new_saver = tf.train.import_meta_graph('./tf_model/model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./tf_model/'))

    tf_dict = {'ipt0:0': x, 'ipt1:0': tb}
    op_to_load = graph.get_tensor_by_name('prediction/BiasAdd:0')    
    
    #uvp
    tout = sess.run(op_to_load, tf_dict)



Ep=5e-4
#plot
def line_plot1():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(x,t,'-og',linewidth=3,label='true')
    plt0, =plt.plot(x,tout[:,0],'r',linewidth=3,label='NN')
    plt.legend(fontsize=20)
    plt.xlabel('x',fontsize=20)
    plt.ylabel('T',fontsize=20)
    #plt.title('%s-u'%(flist[ii]),fontsiuze=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    #plt.savefig('./plot/%s-u'%(flist[ii]), format='png',bbox_inches='tight', dpi=100)
    plt.show() 
    
line_plot1()  
    

#plot
def line_plot2():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(x,ep,'-og',linewidth=3,label='true')
    plt0, =plt.plot(x,tout[:,1]*Ep,'r',linewidth=3,label='NN')
    plt.legend(fontsize=20)
    plt.xlabel('x',fontsize=20)
    plt.ylabel('Ep Vs Al',fontsize=20)
    #plt.title('%s-u'%(flist[ii]),fontsiuze=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    #plt.savefig('./plot/%s-u'%(flist[ii]), format='png',bbox_inches='tight', dpi=100)
    plt.show() 
    
line_plot2() 





