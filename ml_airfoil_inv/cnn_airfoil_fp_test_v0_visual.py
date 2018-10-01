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
path='./airfoil_1600_1aoa_1re/'

data_file='data_sing_144_1600_ts.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
inp_up=result[0]
my_out=result[1]
xx=result[2]
name=result[3]

inp_up=np.asarray(inp_up)
my_out=np.asarray(my_out)
name=np.asarray(name)

xtr1=inp_up[7]
xtr1=np.reshape(xtr1,(1,144,144,1)) 
ttr1=my_out 

inp=xtr1

model=load_model('./hyper_selected/case_144_2/case_sing/final_cnn.hdf5')   

# with a Sequential model
get_out_1c= K.function([model.layers[0].input],
                                  [model.layers[1].output])
c1 = get_out_1c([inp])

get_out_1p= K.function([model.layers[0].input],
                                  [model.layers[2].output])

p1 = get_out_1c([inp])

get_out_2c= K.function([model.layers[0].input],
                                  [model.layers[3].output])
c2 = get_out_2c([inp])

get_out_2p= K.function([model.layers[0].input],
                                  [model.layers[4].output])
p2 = get_out_2p([inp])

get_out_3c= K.function([model.layers[0].input],
                                  [model.layers[5].output])
c3 = get_out_3c([inp])

get_out_3p= K.function([model.layers[0].input],
                                  [model.layers[6].output])
p3 = get_out_3p([inp])

get_out_4c= K.function([model.layers[0].input],
                                  [model.layers[7].output])
c4 = get_out_4c([inp])

get_out_4p= K.function([model.layers[0].input],
                                  [model.layers[8].output])
p4 = get_out_4p([inp])

get_out_5c= K.function([model.layers[0].input],
                                  [model.layers[9].output])
c5 = get_out_5c([inp])

get_out_5p= K.function([model.layers[0].input],
                                  [model.layers[10].output])
p5 = get_out_5p([inp])

get_out_fl= K.function([model.layers[0].input],
                                  [model.layers[11].output])
fl = get_out_fl([inp])

get_out_fc1= K.function([model.layers[0].input],
                                  [model.layers[12].output])
fc1 = get_out_fc1([inp])

get_out_fc2= K.function([model.layers[0].input],
                                  [model.layers[13].output])
fc2 = get_out_fc2([inp])

get_out_fc3= K.function([model.layers[0].input],
                                  [model.layers[14].output])
fc3 = get_out_fc3([inp])

get_out_fc4= K.function([model.layers[0].input],
                                  [model.layers[15].output])
fc4 = get_out_fc4([inp])


#out=model_test.predict([xtr1])
#out=out*0.18

#plot one CNN
#out=c5[0][0,:,:,:8]
out=xtr1[0,:,:,:]
for k in range(1):
    
    plt.figure(figsize=(3, 3), dpi=100)
    xp, yp = np.meshgrid(range(out.shape[0]), range(out.shape[1]))
    
    cp=plt.imshow(out[:,:,k])
    plt.axis('off')
       
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.00, wspace = 0)
    plt.savefig('./plot1/c00_%d.eps'%(k), format='eps', dpi=100)
    plt.show()
    

'''#plot multiple
for k in range(10):

    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx,my_out[k][0:35],'o',mfc='grey',mec='grey',ms=10,label='true')
    plt.plot(xx,my_out[k][35:],'o',mfc='grey',mec='grey',ms=8,)
    
    #plt.plot(xx,out1[k][0:35],'k',lw=2,label='CNN-1')
    #plt.plot(xx,out1[k][35:],'k',lw=2)
    
    #plt.plot(xx,out2[k][0:35],'b',lw=2,label='CNN-2')
    #plt.plot(xx,out2[k][35:],'b',lw=2)
    
    plt.plot(xx,out3[k][0:35],'k',lw=3,label='CNN-3')
    plt.plot(xx,out3[k][35:],'k',lw=3)
    
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.2,0.2])
    plt.legend(loc="upper left", bbox_to_anchor=[0.1, 1], ncol=2, fontsize=16, \
               frameon=False, shadow=False, fancybox=False,title='')
    plt.xlabel('X',fontsize=20)
    plt.ylabel('Y',fontsize=20)  
    #plt.axis('off')
    plt.tight_layout()
    plt.savefig('./plot/tr_%s_%s.eps'%(k,name[k]), format='eps', bbox_inches='tight',dpi=100)
    plt.show()'''




