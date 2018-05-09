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


path='./naca456'

indir=path+"/coord"
fname = [f for f in listdir(indir) if isfile(join(indir, f))]

#load coord
img_mat=[]
for i in range(len(fname)):
#for i in range(1):
    tmp_co=np.loadtxt(indir+'/%s'%fname[i],skiprows=1)
    
    #plot
    figure=plt.figure(figsize=(3,3))
    plt0, =plt.plot(tmp_co[:,0],tmp_co[:,1],'k',linewidth=0.1,label='true')
    #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
    #plt.legend(fontsize=16)
    #plt.xlabel('alpha',fontsize=16)
    #plt.ylabel('cl',fontsize=16)
    #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    plt.xlim(0,1.)
    plt.ylim(-0.18,0.18)    
    plt.axis('off')
    #plt.grid(True)
    #patch.set_facecolor('black')
    plt.savefig('./coord_plot/%s'%fname[i], format='png')
    plt.show() 

    img = io.imread('./coord_plot/%s'%fname[i], as_grey=True)  # load the image as grayscale
    img = util.invert(img)
    img_mat.append(img)
    print 'image matrix size: ', img.shape      # print the size of image
    #print '\n First 5 columns and rows of the image matrix: \n', img[150:210,170:180] 
    #viewer.ImageViewer(img).show()  
    #img=img-1
    #img=abs(img)
    #viewer.ImageViewer(img).show()



data1=[img_mat,fname]
with open(path+'/data_airfoil_inverse.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

