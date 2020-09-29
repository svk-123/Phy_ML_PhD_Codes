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

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

import os, shutil
from skimage import io, viewer,util 

# ref:[data,name]
path='./'

indir=path+"polar"
fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()


#load CL CD
name=[]   
rey_no=[]
data1=[]  
for i in range(len(fname)):
    
    tmp_data=np.loadtxt(indir+'/%s'%fname[i],delimiter=',',skiprows=10)  
     
    if len(tmp_data != 0):
        with open(indir+'/%s'%fname[i],'r') as myfile:
            data0=myfile.readlines()
        
            if "Calculated polar for:" in data0[2]:
                    name.append(data0[2].split(":",1)[1].strip()) 
        
            if "Re =" in data0[7]:
                    tmp=data0[7].split("Re =",1)[1]
                    rey_no.append(tmp.split("e",1)[0])
        
        #load alpha cl cd
        tmp_data=np.loadtxt(indir+'/%s'%fname[i],delimiter=',',skiprows=10)
        data1.append(tmp_data[0:3])

rey_no=np.asarray(rey_no)
#alpha cl cd
data1=np.asarray(data1)
        


img_mat=[]
for i in range(len(name)):
    print i
    coord=np.loadtxt('./coord_seligFmt_formatted/%s.dat'%name[i],skiprows=1)
    #plot
    figure=plt.figure(figsize=(2,2))
    plt0, =plt.plot(coord[:,0],coord[:,1],'k',linewidth=0.5,label='true')
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.18,0.18)    
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.savefig('./plot/%s.eps'%name[i], format='eps')
    plt.close() 

    img = io.imread('./plot/%s.eps'%name[i], as_grey=True)  # load the image as grayscale
    img = util.invert(img)
    img_mat.append(img)
    print 'image matrix size: ', img.shape      # print the size of image


info=['img,clcd,name']
dataf=[img_mat,data1[:,1:3],name,info]
with open('./data_file/data_clcd.pkl', 'wb') as outfile:
    pickle.dump(dataf, outfile, pickle.HIGHEST_PROTOCOL)







