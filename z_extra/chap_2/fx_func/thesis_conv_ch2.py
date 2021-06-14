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
from os import listdir
from os.path import isfile, join
import sys

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers import merge, Input, dot, add, concatenate
import random

from keras.models import model_from_json
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import cPickle as pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten,UpSampling2D
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K
from numpy import linalg as LA
import os, shutil

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
#plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rc('font', family='serif')

#matplotlib.rcParams["font.family"] = "Times"
#matplotlib.rc('font',**{'family':'serif','serif':['Times']})
#matplotlib.rc('text', usetex=True)
# u'LMRoman10'


"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

'''
l1=120
l2=150
l3=200

#CNN1- fc layers
plt.figure(figsize=(6,5),dpi=100)
path='./data_file/'

with open('./hist/hist_l2_30_tanh.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]
h1l=h1['loss']
h1vl=h1['val_loss']
h1l=np.asarray(h1l)
h1vl=np.asarray(h1vl)

with open('./selected_model/mlp_4x30/hist/hist_l4_30_tanh.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]
h2l=h2['loss']
h2vl=h2['val_loss']
h2l=np.asarray(h2l)
h2vl=np.asarray(h2vl)

with open('./selected_model/mlp_4x50/hist/hist_l4_50_tanh.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]
h3l=h3['loss']
h3vl=h3['val_loss']
h3l=np.asarray(h3l)
h3vl=np.asarray(h3vl)

with open('./selected_model/mlp_6x50/hist/hist_l6_50_tanh.pkl', 'rb') as infile:
    result = pickle.load(infile)
h4=result[0]
h4l=h4['loss']
h4vl=h4['val_loss']
h4l=np.asarray(h4l)
h4vl=np.asarray(h4vl)
'''

#with open('../selected_model/case_1_p16_tanh_C5F9/model_cnn/hist.pkl', 'rb') as infile:
#    result = pickle.load(infile)
#h5=result[0]
#h5l=h5['loss']
#h5vl=h5['val_loss']
#h5l=np.asarray(h5l)
#h5vl=np.asarray(h5vl)
#
#with open('../selected_model/case_1_p6_tanh_C5F7/model_cnn/hist.pkl', 'rb') as infile:
#    result = pickle.load(infile)
#h6=result[0]
#h6l=h6['loss']
#h6vl=h6['val_loss']
#h6l=np.asarray(h6l)
#h6vl=np.asarray(h6vl)
#
#with open('../selected_model/case_1_p8_tanh_C5F7/model_cnn/hist.pkl', 'rb') as infile:
#    result = pickle.load(infile)
#h7=result[0]
#h7l=h7['loss']
#h7vl=h7['val_loss']
#h7l=np.asarray(h7l)
#h7vl=np.asarray(h7vl)


'''
l1=150
l2=200
l3=250

plt.figure(figsize=(6,5),dpi=100)
path='./data_file/'

with open('./selected_model/mlp_2x30_sigmoid/hist/hist_l4_30_tanh.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]
h1l=h1['loss']
h1vl=h1['val_loss']
h1l=np.asarray(h1l)
h1vl=np.asarray(h1vl)

with open('./selected_model/mlp_2x30/hist/hist_l2_30_tanh.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]
h2l=h2['loss']
h2vl=h2['val_loss']
h2l=np.asarray(h2l)
h2vl=np.asarray(h2vl)

with open('./selected_model/mlp_2x30_relu/hist/hist_l4_30_tanh.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]
h3l=h3['loss']
h3vl=h3['val_loss']
h3l=np.asarray(h3l)
h3vl=np.asarray(h3vl)

plt.plot(range(len(h1l)),h1l,'r',marker='v',mfc='r',ms=6,lw=2,markevery=l1,label='Sigmoid')
plt.plot(range(len(h2l)),h2l,'b',marker='o', mfc='b',ms=6,markevery=l2,lw=2,label='Tanh')
plt.plot(range(len(h3l)),h3l,'g',lw=2,marker='^', mfc='g',ms=6,markevery=l3,label='ReLU')
#plt.plot(range(len(h4l)),h4l,'c',lw=2,marker='<', mfc='c',ms=12,markevery=l3,label='MLP 6x50 tr')

#plt.plot(range(len(h1vl)),h1vl, 'r',marker='v',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='MLP 1x30 val')
#plt.plot(range(len(h2vl)),h2vl, 'b',marker='o',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='MLP 4x30 val')
#plt.plot(range(len(h3vl)),h3vl, 'g',marker='^',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='MLP 4x50 val')
#plt.plot(range(len(h3vl)),h3vl, 'c',marker='<',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='MLP 6x50 val')
    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
#plt.figtext(0.45, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))

plt.xlim([-50,3000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('conv_acti.tiff', format='tiff', bbox_inches='tight',dpi=300)
plt.show()
'''


l1=150
l2=200
l3=250

plt.figure(figsize=(6,5),dpi=100)
path='./data_file/'

with open('./selected_model/mlp_2x30/hist/hist_l2_30_tanh.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]
h1l=h1['loss']
h1vl=h1['val_loss']
h1l=np.asarray(h1l)
h1vl=np.asarray(h1vl)

with open('./selected_model/mlp_2x45/hist/hist_l2_45_tanh.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]
h2l=h2['loss']
h2vl=h2['val_loss']
h2l=np.asarray(h2l)
h2vl=np.asarray(h2vl)

with open('./selected_model/mlp_3x30/hist/hist_l3_30_tanh.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]
h3l=h3['loss']
h3vl=h3['val_loss']
h3l=np.asarray(h3l)
h3vl=np.asarray(h3vl)

plt.plot(range(len(h1l)),h1l,'r',marker='v',mfc='r',ms=6,lw=2,markevery=l1,label='2 x 30')
plt.plot(range(len(h2l)),h2l,'b',marker='o', mfc='b',ms=6,markevery=l2,lw=2,label='2 x 45')
plt.plot(range(len(h3l)),h3l,'g',lw=2,marker='^', mfc='g',ms=6,markevery=l3,label='3 x 30')
#plt.plot(range(len(h4l)),h4l,'c',lw=2,marker='<', mfc='c',ms=12,markevery=l3,label='MLP 6x50 tr')

#plt.plot(range(len(h1vl)),h1vl, 'r',marker='v',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='MLP 1x30 val')
#plt.plot(range(len(h2vl)),h2vl, 'b',marker='o',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='MLP 4x30 val')
#plt.plot(range(len(h3vl)),h3vl, 'g',marker='^',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='MLP 4x50 val')
#plt.plot(range(len(h3vl)),h3vl, 'c',marker='<',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='MLP 6x50 val')
    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
#plt.figtext(0.45, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))

plt.xlim([-50,3000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('conv_hyp.tiff', format='tiff', bbox_inches='tight',dpi=300)
plt.show()

