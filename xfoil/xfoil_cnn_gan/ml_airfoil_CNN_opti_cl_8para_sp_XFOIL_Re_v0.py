#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

with new parameters tanh_16_v1:

with new flow prediction network using v1.

@author: vinoth
"""
#based on parameters 
from __future__ import division
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
from keras.layers import merge, Input, dot

import random

from keras.models import model_from_json
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

from numpy import linalg as LA
import os, shutil
from scipy.optimize import minimize

##Xfoil imports
from kulfan_to_coord import CST_shape
import argparse
from utils import mean_err
from simulation import evaluate,compute_coeff
####------------


import matplotlib
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 


# ref:[data,name]
folder = './tmp1/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)


cd=[]
cl=[]
mypara=[]
name=[]
xx_=[]

#load airfoil para
path='./model_cnn_v1/'
data_file='cnn_uiuc_para_8_tanh_v1.pkl'
with open(path + data_file, 'rb') as infile:
    result2 = pickle.load(infile)

mypara.extend(result2[0])
name.extend(result2[1])
xx_.extend(result2[2])

mypara=np.asarray(mypara)
name=np.asarray(name)
xx_=np.asarray(xx_)
xx_=xx_[:101]

#nname=[]
#for i in range(len(name)):
#    nname.append(name[i].decode())
#name=np.asarray(nname)

#fp=open('xx_101.txt','w')
#for i in range(len(xx_)):
#    fp.write('%f \n'%xx_[i])
#    
#fp.close()


model_para=load_model('./model_cnn_v1/case_1_p8_tanh_include_naca4_v2/model_cnn/final_cnn.hdf5') 
get_c= K.function([model_para.layers[16].input],  [model_para.layers[19].output])

global scaler

global tar_cl
global init_cl
global reno
global aoa
global mach
global xx

xx=xx_

tar_cl=1.0
init_cl=0
reno=100000.0
mach=0.0
aoa_st=np.asarray([0])

reno=np.asarray(reno)
aoa_st=np.asarray(aoa_st)

global my_counter
my_counter =0

global error
error=[]

def get_coord(p2):
    
    para1=p2[:8]
    para1=np.reshape(para1,(1,8))
    c1 = get_c([para1])[0][0,:]
    c1=c1*0.2
    c2=np.concatenate((c1[0:100],-c1[0:1]),axis=0)
    return (np.asarray([xx,c2]).transpose())


#### check airfoil plot#####
#fn='n0012'
#idx=np.argwhere(name=='%s'%fn)
##scaled parameter
#p1=mypara[idx[0][0],:]
#xy=get_coord(p1)
#plt.figure(figsize=(6,5),dpi=100)
#plt.plot(xy[:,0],xy[:,1],'r')
#plt.ylim([-0.5,0.5])
#plt.show()
#plt.close()
#########...................


def loss(para):
       
    global my_counter  
    global init_cl
    global pred_cl
    mypara=para[:8]
    aoa=para[8]*8.0
    
    xy=get_coord(mypara)
                   
    _,pred_cl,pred_cd=evaluate(xy,reno,mach,aoa,200,True)
    
    if(pred_cl > 0):
        e=(pred_cd/pred_cl)*100
    else:
        e=100.0
    
#    if(pred_cl > tar_cl):
#        #e=np.sqrt(((tar_cl - pred_cl) ** 2))
#        e=0
#    else:
#        e=np.sqrt(((tar_cl - pred_cl) ** 2))
     
    
    fp.write('%s %s %s\n'%(my_counter,e,pred_cl))    
    if(my_counter == 0):
        init_cl=pred_cl
    my_counter = my_counter +1
    print ('Iter: %d, E: %f , Pred_cl: %f,aoa: %f' %(my_counter,e,pred_cl/pred_cd,aoa))
    
     
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xy[:,0],xy[:,1],'r')
    plt.ylim([-0.3,0.3])
    plt.savefig('./tmp1/%s.png'%my_counter,format='png')
    plt.close()
                  
    return  e
     

#base foil name
np.random.seed(123)    
for iters in range(1):
    aa=mypara.copy()
    #base foil name
    idx1=np.random.randint(1425)   
    fn=name[idx1]
    fn='n0012'
    print(fn)
    idx=np.argwhere(name=='%s'%fn)    
    
    #scaled parameter
    p1_=mypara[idx[0][0],:]
    p1=np.concatenate((p1_,aoa_st),axis=0)
    #conv file
    fp=open('./opt_plot/conv_%s.dat'%fn,'w+')
     
    print('Starting loss = {}'.format(loss(p1)))
    print('Intial foil = %s' %name[idx[0]])
    a1=-0.2
    a2=0.2
    mylimit=((aa[:,0].min()+a1,aa[:,0].max()+a2),(aa[:,1].min()+a1,aa[:,1].max()+a2),(aa[:,2].min()+a1,aa[:,2].max()+a2),\
             (aa[:,3].min()+a1,aa[:,3].max()+a2),(aa[:,4].min()+a1,aa[:,4].max()+a2),(aa[:,5].min()+a1,aa[:,5].max()+a2),\
             (aa[:,6].min()+a1,aa[:,6].max()+a2),(aa[:,7].min()+a1,aa[:,7].max()+a2),(0,1))

    res = minimize(loss, x0=p1, method = 'L-BFGS-B', bounds=mylimit, \
                   options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
                                     'eps': 0.05, 'maxfun': 100, \
                                     'maxiter': 100, 'maxls': 100})
    #res = minimize(loss, x0=p1, method = 'L-BFGS-B', \
    #               options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
    #                                 'eps': 0.01, 'maxfun': 100, \
    #                                 'maxiter': 50, 'maxls': 100})
        
    
    print('Ending loss = {}'.format(loss(res.x)))
    fp.close()
    
    fp=open('./opt_plot/final_%s.dat'%fn,'w')
    xy=get_coord(res.x[:8])
    for i in range(len(xy)):
        fp.write('%f %f 0.00\n'%(xy[i,0],xy[i,1]))
    fp.close()
    
    fp=open('./opt_plot/resx_%s.dat'%fn,'w')
    fp.write('%s'%res.x)
    fp.close()
    
    #intial shape
    xy0=get_coord(p1)
    
    
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xy0[:,0],xy0[:,1],'--k',label='Base')
    plt.plot(xy[:,0],xy[:,1],'g',lw=3,label='Optimized')
    plt.legend(fontsize=14,fancybox=False,frameon=False, shadow=False)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.25,0.25])
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)
    plt.savefig('./opt_plot/opti_%s.png'%fn,format='png',bbox_inches='tight',dpi=300)
    plt.close()
