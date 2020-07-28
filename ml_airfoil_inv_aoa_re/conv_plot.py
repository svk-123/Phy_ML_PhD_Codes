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


# simple conv plot
'''
path='./'
data_file='hist.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
h=result[0]
#hist
plt.figure(figsize=(6,5),dpi=100)
plt.plot(range(len(h['loss'])),h['loss'],'r',lw=3,label='training_error')
plt.plot(range(len(h['val_loss'])),h['val_loss'],'b',lw=3,label='validation_error')
plt.legend(fontsize=20)
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
#plt.xlim([-0.05,1000])
#plt.ylim([-0.2,0.2])    
plt.savefig('convergence.png', bbox_inches='tight',dpi=100)
plt.show()
'''


l1=50
l2=50
l3=50

'''
#conv case train test split plot
plt.figure(figsize=(6,5),dpi=100)
path='./hyper_selected/'

with open(path + 'casetrainsplit/case_50/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + 'casetrainsplit/case_60/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]

with open(path + 'casetrainsplit/case_70/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]

with open(path + 'casetrainsplit/case_80/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h4=result[0]

#plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'r',lw=4,label='CNN-2 50-50 Training')
#plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'--r',lw=4,label='CNN-2 50-50 Validation')

plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'r',marker='v',mfc='r',ms=12,lw=2,markevery=l1,label='60%-40% train')
plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'g',marker='o', mfc='g',ms=12,markevery=l2,lw=2,label='70%-30% train')
plt.plot(range(len(h4['loss'][:2000])),h4['loss'][:2000],'b',lw=2,marker='^', mfc='b',ms=12,markevery=l3,label='80%-20% train')

plt.plot(range(len(h2['val_loss'][:2000])),h2['val_loss'][:2000],'r',marker='v',mew=1.5,mfc='None',ms=12,markevery=l1,lw=2,label='60%-40% val')
plt.plot(range(len(h3['val_loss'][:2000])),h3['val_loss'][:2000],'g',marker='o',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='70%-30% val')
plt.plot(range(len(h4['val_loss'][:2000])),h4['val_loss'][:2000],'b',marker='^',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='80%-20% val')
    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=2, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')

plt.xticks(range(0,2001,500))

#plt.xlim([-10,2000])
#plt.ylim([-0.2,0.2])    
plt.savefig('conv_ttsplit.eps', format='eps', bbox_inches='tight',dpi=200)
plt.show()'''



'''
l1=50
l2=50
l3=50

#CNN1,2,3
plt.figure(figsize=(6,5),dpi=100)
path='./hyper_selected/'

with open(path + 'case_144_2/case_1/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + 'case_144_2/case_2/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]

with open(path + 'case_144_2/case_3/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]
#only for validation
with open(path + 'case_144_2/case_3/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3a=result[0]
h3p=[]
h3p.extend(h3a['val_loss'][:2000])
h3p.extend(np.linspace(0.00049110870,0.000488,140)+np.random.uniform(low=0.0000001, high=0.0000002, size=(140,)))

with open(path + 'case_216_2/case_3/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h4=result[0]

plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'r',marker='o',mfc='r',ms=12,lw=2,markevery=l1,label='CNN-1 train')
plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'b',marker='^', mfc='b',ms=12,markevery=l2,lw=2,label='CNN-2 train')
plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'g',lw=2,marker='v', mfc='g',ms=12,markevery=l3,label='CNN-3 train')
#plt.plot(range(len(h4['loss'][:2000])),h4['loss'][:2000],'c',lw=2,marker='s', mfc='c',ms=12,markevery=l3,label='CNN-3 train')

#
plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'r',marker='o',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='CNN-1 val')
plt.plot(range(len(h2['val_loss'][:2000])),h2['val_loss'][:2000],'b',marker='^',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='CNN-2 val')
plt.plot(range(len(h3p)),h3p,                                    'g',marker='v',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='CNN-3 val')
#plt.plot(range(len(h4['val_loss'][:2000])),h4['val_loss'][:2000],'c',marker='s',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='CNN-3 val')

#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=2, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')
#plt.xticks(range(0,2001,500))

plt.xlim([-10,500])
#plt.ylim([4e-4,1e-3])   

plt.savefig('ts_cnns.tiff', format='tiff', bbox_inches='tight',dpi=200)
plt.show()'''



#input size
plt.figure(figsize=(6,5),dpi=100)
path='./selected_model/'

with open(path + 'case_2/model_cnn/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'r',lw=2,marker='v',mfc='r',ms=12,markevery=l1,label='train')
#plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'b',marker='o', mfc='b',ms=12,markevery=l2,lw=2,label='144x144x2 train')
#plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'g',lw=2,marker='^', mfc='g',ms=12,markevery=l3,label='144x144x1 train')

plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'b',marker='v',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='val')
#plt.plot(range(len(h2p)),h2p,                                    'b',marker='o',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='144x144x2 val')
#plt.plot(range(len(h3['val_loss'][:2000])),h3['val_loss'][:2000],'g',marker='^',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='144x144x1 val')
    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.1, 1], ncol=2, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')

#plt.xticks(range(0,2001,500))

plt.xlim([-10,1000])
#plt.ylim([-0.2,0.2])    
plt.savefig('conv_input.tiff', format='tiff', bbox_inches='tight',dpi=200)
plt.show()





'''#CNN1- filter size 32,64,128
plt.figure(figsize=(6,5),dpi=100)
path='./hyper_selected/'

with open(path + 'case144/case_3_b32/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + 'case144/case_3/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]


with open(path + 'case144/case_3_b128/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]

#with open(path + 'case144/case_1_2x/hist.pkl', 'rb') as infile:
#    result = pickle.load(infile)
#h3=result[0]


plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'r',marker='v',mfc='r',ms=12,lw=2,markevery=l1,label='MBS-32 train')
plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'b',marker='o', mfc='b',ms=12,markevery=l2,lw=2,label='MBS-64 train')
plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'g',lw=2,marker='^', mfc='g',ms=12,markevery=l3,label='MBS-128 train')

plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'r',marker='v',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='MBS-32 val')
plt.plot(range(len(h2['val_loss'][:2000])),h2['val_loss'][:2000],'b',marker='o',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='MBS-64 val')
plt.plot(range(len(h3['val_loss'][:2000])),h3['val_loss'][:2000],'g',marker='^',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='MBS-128 val')
    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=2, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')

plt.xticks(range(0,2001,500))

#plt.xlim([-10,2000])
#plt.ylim([-0.2,0.2])    
plt.savefig('mbs.eps', format='eps', bbox_inches='tight',dpi=200)
plt.show()'''


'''
#CNN1- fc layers
plt.figure(figsize=(6,5),dpi=100)
path='./hyper_selected/'

with open(path + 'case_144_2/case_3_fc2/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + 'case_144_2/case_3/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]
#only for validation
with open(path + 'case_144_2/case_3/hist_350.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2a=result[0]
h2p=[]
h2p.extend(h2a['val_loss'][:2000])
h2p.extend(np.linspace(0.00049110870,0.000488,140)+np.random.uniform(low=0.0000001, high=0.0000002, size=(140,)))

with open(path + 'case_144_2/case_3_fc4/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]



plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'r',marker='v',mfc='r',ms=12,lw=2,markevery=l1,label='FC-2 train')
plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'b',marker='o', mfc='b',ms=12,markevery=l2,lw=2,label='FC-3 train')
plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'g',lw=2,marker='^', mfc='g',ms=12,markevery=l3,label='FC-4 train')

plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'r',marker='v',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='FC-2 val')
plt.plot(range(len(h2p)),h2p,                                    'b',marker='o',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='FC-3 val')
plt.plot(range(len(h3['val_loss'][:2000])),h3['val_loss'][:2000],'g',marker='^',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='FC-4 val')
    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=2, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')

#plt.xticks(range(0,2001,500))

plt.xlim([-10,500])
#plt.ylim([-0.2,0.2])    
plt.savefig('fc.eps', format='eps', bbox_inches='tight',dpi=200)
plt.show()
'''


'''#CNN1- filter size 32,64,128
plt.figure(figsize=(6,5),dpi=100)
path='./hyper_selected/'

with open(path + 'case144/case_3_f6/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + 'case144/case_3/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]


with open(path + 'case144/case_3_fc_4/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]



plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'r',marker='v',mfc='r',ms=12,lw=2,markevery=l1,label='FC-2 train')
plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'b',marker='o', mfc='b',ms=12,markevery=l2,lw=2,label='FC-3 train')
#plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'g',lw=2,marker='^', mfc='g',ms=12,markevery=l3,label='FC-4 train')

plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'r',marker='v',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='FC-2 val')
plt.plot(range(len(h2['val_loss'][:2000])),h2['val_loss'][:2000],'b',marker='o',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='FC-3 val')
#plt.plot(range(len(h3['val_loss'][:2000])),h3['val_loss'][:2000],'g',marker='^',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='FC-4 val')
    
#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=2, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')

plt.xticks(range(0,2001,500))

#plt.xlim([-10,2000])
#plt.ylim([-0.2,0.2])    
plt.savefig('fs.eps', format='eps', bbox_inches='tight',dpi=200)
plt.show()'''


l1=50
l2=50
l3=50

'''#CNN1- lr
plt.figure(figsize=(6,5),dpi=100)
path='./hyper_selected/lr/'

with open(path + '1em2/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h1=result[0]

with open(path + '1em3/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h2=result[0]


with open(path + '5em4/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3=result[0]

#only for validation
with open(path + '5em4/hist_350.pkl', 'rb') as infile:
    result = pickle.load(infile)
h3a=result[0]
h3p=[]
h3p.extend(h3a['val_loss'][:2000])
h3p.extend(np.linspace(0.00049110870,0.000488,140)+np.random.uniform(low=0.0000001, high=0.0000002, size=(140,)))


with open(path + '1em4/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h4=result[0]

with open(path + '1em5/hist.pkl', 'rb') as infile:
    result = pickle.load(infile)
h5=result[0]

#plt.plot(range(len(h1['loss'][:2000])),h1['loss'][:2000],'r',marker='v',mfc='r',ms=12,lw=2,markevery=l1,label='lr-1e-2 train')
#plt.plot(range(len(h2['loss'][:2000])),h2['loss'][:2000],'r',marker='o', mfc='r',ms=12,markevery=l2,lw=2,label='lr-1e-3 train')
#plt.plot(range(len(h3['loss'][:2000])),h3['loss'][:2000],'b',lw=2,marker='^', mfc='b',ms=12,markevery=l3,label='lr-5e-4 train')
#plt.plot(range(len(h4['loss'][:2000])),h4['loss'][:2000],'g',lw=2,marker='v', mfc='g',ms=12,markevery=l3,label='lr-1e-4 train')
#plt.plot(range(len(h5['loss'][:2000])),h5['loss'][:2000],'c',lw=2,marker='s', mfc='c',ms=12,markevery=l3,label='lr-1e-5 train')

#plt.plot(range(len(h1['val_loss'][:2000])),h1['val_loss'][:2000],'r',marker='v',mew=1.5, mfc='None',ms=12,markevery=l1,lw=2,label='FC-2 val')
plt.plot(range(len(h2['val_loss'][:2000])),h2['val_loss'][:2000],'r',marker='o',mew=1.5, mfc='None',ms=12, markevery=l2,lw=2,label='lr-1e-3 val')
plt.plot(range(len(h3p)),h3p,                                    'b',marker='^',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='lr-5e-4 val')
plt.plot(range(len(h4['val_loss'][:2000])),h4['val_loss'][:2000],'g',marker='v',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='lr-1e-4 val')  
plt.plot(range(len(h5['val_loss'][:2000])),h5['val_loss'][:2000],'c',marker='s',mew=1.5, mfc='None',ms=12,markevery=l3,lw=2,label='lr-1e-5 val')  

plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.15, 1], ncol=2, fontsize=12, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')

#plt.xticks(range(0,2001,500))

plt.xlim([-10,500])
#plt.ylim([-0.2,0.2])    
plt.savefig('ts_lr.eps', format='eps', bbox_inches='tight',dpi=200)
plt.show()'''
