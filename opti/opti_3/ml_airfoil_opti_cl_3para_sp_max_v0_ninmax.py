#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

with new parameters tanh_16_v1:

with new flow prediction network using v1.

maximize cl , while contrain Cd


@author: vinoth
"""
#based on parameters 

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
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

from numpy import linalg as LA
import os, shutil
from skimage import io, viewer,util 
from scipy.optimize import minimize

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

from naca import naca4

# ref:[data,name]
#load airfoil para
path='./data_file/'
data_file='naca4_clcd_turb_st_3para.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)

cd=result1[1]
cl=result1[2]
myreno=result1[3]
myaoa=result1[4]
mypara=result1[5]
name=result1[6]

mypara=np.asarray(mypara)
name=np.asarray(name)

#nname=[]
#for i in range(len(name)):
#    nname.append(name[i].decode())
#name=np.asarray(nname)


#fp=open('clcd_data_turb_st.txt','w')
#for i in range(len(cl)):
#    fp.write('%s %f %f %f %f\n'%(name[i],myreno[i],myaoa[i],cl[i],cd[i]))
#    
#fp.close()
    

#load model para: airfoil coord from parameters
#scaler used in cl pred network
# reno=10000
#aoa=14
#para=np.array([6,6,30])
#cl=0.8
#cd=0.25

model=load_model('./selected_model_0p9/turb_naca4_3para_st_6x80_tanh_minmax/model/final_sf.hdf5')  
global scaler
scaler=np.array([6,6,30])

global tar_cl
global reno
global aoa
global init_cl
global final_cd
global tar_cd

path='./result_paper_v7/sp_1_tanh_cdc/'
tar_cl=1.2
tar_cd=0.045

reno=np.asarray([20000])/100000.
aoa=np.asarray([4])/14.

reno=np.asarray(reno)
aoa=np.asarray(aoa)

np.random.seed(12343)
#np.random.seed(12345)
I=range(481)
np.random.shuffle(I)
#mp2-relu
#foil=['naca5014','naca5008','naca3012','naca2030','naca2010','naca1416']
foil=np.unique(name)[I[0:10]]

for jj in range(len(foil)):
        
    global my_counter
    my_counter =0
    
    global error
    error=[]
    
    def get_coord(p2):
        x,y=naca4(p2*scaler,100)
        return (x,y)
    
    def loss(para):
           
        global my_counter  
        global init_cl
        global final_cl
        global final_cd
        global tar_cd
        maxi=1.0
        mini=-1.0
        
        mypara=para
    
        x,y=get_coord(mypara)
    
        my_inp=np.concatenate((reno,aoa,mypara),axis=0)
        my_inp=np.reshape(my_inp,(1,5))
        #cd, cl
        n_out=model.predict([my_inp])
        
        pred_cd = ((0.25-0.01)/(maxi-mini))*(n_out[0,0]-maxi)  + 0.25
        pred_cl = ((1.45+0.35)/(maxi-mini))*(n_out[0,1]-maxi)  + 1.45
               
        
        print ('Pred_cl:', pred_cl, pred_cd)

        #e= (1.0/pred_cl)
        
        e= abs(pred_cd-tar_cd)*10.0 + (1.0/pred_cl)
        

        print ('mse:', e)
        
        final_cd = pred_cd
        final_cl = pred_cl
        
        if(my_counter == 0):
            init_cl=pred_cl
            fp_conv.write('init-Cl = %s\n'%(init_cl)) 
            fp_conv.write('Iter,   MSE,    Pred_cl\n')
            
        my_counter = my_counter +1
        print ('Iter:', my_counter)
        
        fp_conv.write('%s %s %s\n'%(my_counter,e,pred_cl)) 
           
        plt.figure(figsize=(6,5),dpi=100)
        plt.plot(x,y,'r',label='true')
        plt.ylim([-0.5,0.5])
        plt.savefig('./opt_plot/%s.png'%my_counter,format='png')
        plt.close()
           
        
            
        return  e
    
    #naca2412, 4510,3310,
    fn=foil[jj]     

    
    
    fp_conv=open(path+'conv_%s.dat'%fn,'w+')
    
    idx=np.argwhere(name=='%s'%fn)
    
    #naca4510
    p1=mypara[idx[0][0],:]/scaler
    
     
    print('Starting loss = {}'.format(loss(p1)))
    print('Intial foil = %s' %name[idx[0]])
    
    mylimit=((0,1.1),(0,1.1),(0.2,1.1))
    #mylimit=((0,6),(0,6),(6,32))
    res = minimize(loss, x0=p1, method = 'L-BFGS-B', bounds=mylimit, \
                   options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
                                     'eps': 0.001, 'maxfun': 100, \
                                     'maxiter': 100, 'maxls': 100})
    
      
    print('Ending loss = {}'.format(loss(res.x)))
    
    fp=open(path+'final_%s.dat'%fn,'w')
    x,y=get_coord(res.x)
    for i in range(len(x)):
        fp.write('%f %f 0.00\n'%(x[i],y[i]))
    fp.close()
    
    fp=open(path+'base_%s.dat'%fn,'w')
    x00,y00=get_coord(p1)
    for i in range(len(x00)):
        fp.write('%f %f 0.00\n'%(x00[i],y00[i]))
    fp.close()
    
    fp=open(path+'resx_%s.dat'%fn,'w')
    fp.write('Re = %s\n'%(reno*100000))
    fp.write('AoA = %s\n'%(aoa*14))
    fp.write('init-Cl = %s\n'%(init_cl))
    fp.write('tar-Cl = %s\n'%(tar_cl))
    fp.write('final-Cl = %s\n'%(final_cl))
    fp.write('final-Cd = %s\n'%(final_cd))     
    fp.write('%s\n'%res.x)
    fp.write('%s\n'%(res.x*[6,6,30]))
    fp.close()
    
    #intial shape
    x0,y0=get_coord(p1)
    
    
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(x0,y0,'--k',label='Base')
    plt.plot(x,y,'g',lw=3,label='Optimized')
    plt.legend(fontsize=14)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.25,0.25])
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)
    plt.savefig(path+'opti_%s.png'%fn,bbox_inches='tight',dpi=300)
    plt.close()
    
    
    fp_conv.close()  