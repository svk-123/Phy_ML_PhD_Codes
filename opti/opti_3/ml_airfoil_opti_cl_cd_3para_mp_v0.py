#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

with new parameters tanh_16_v1:

with new flow prediction network using v1.

cl by cd network is used

multi-point opti is used

pred_cl is cl by cd here

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
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

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

#fp=open('cl_data.txt','w')
#for i in range(len(cl)):
#    fp.write('%s %f %f %f \n'%(name[i],myreno[i],myaoa[i],cl[i]))
#    
#fp.close()
    

#load model para: airfoil coord from parameters
#scaler used in cl pred network
# reno=10000
#aoa=14
#para=np.array([6,6,30])
#cl=0.8
#cd=0.25
model=load_model('./selected_model_0p9/turb_naca4_3para_st_6x80_tanh_clbycd/model/final_sf.hdf5')  
path='./result_paper_v6/mp_1_tanh/'


global scaler
scaler=np.array([6,6,30])

global tar_cl
global pred_cl
global reno
global aoa
global init_cl

#mp-1
tar_cl=np.asarray([1.0094,1.121,1.1463,1.1647])
tar_cd=np.asarray([0.054,0.046,0.044,0.044])

pred_cl=np.asarray([0,0,0,0])
init_cl=np.asarray([0,0,0,0])

pred_cd=np.asarray([0,0,0,0])
init_cd=np.asarray([0,0,0,0])

reno=np.asarray([10000,20000,30000,40000])/100000.
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
         
    
    def get_coord(p2):
        x,y=naca4(p2*scaler,100)
        return (x,y)
    
    def loss(para):
           
        global my_counter  
        global pred_cl
        global init_cl
        global pred_cd
        global init_cd        
        para=para
    
        x,y=get_coord(para)
    
    
        inp_aoa=np.repeat(aoa, len(reno)) 
        inp_para=np.repeat(para[:,None],len(reno),axis=1).transpose() 
        
        my_inp=np.concatenate((reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
            
        #cd, cl
        out=model.predict([my_inp])
            
        out=out*32.0
                           
        pred_cl=out[:,0]
        
        e=1.0/np.sqrt(sum(pred_cl**2))
        
        print ('Pred_cl:', pred_cl)
               
        print ('mse:', e)
        
        fp_conv.write('%s %s %s \n'%(my_counter,e,pred_cl))    
        if(my_counter == 0):
            init_cl=pred_cl
            init_cd=pred_cd            
            fp_conv.write(' Iter, MSE, Pred_cl \n') 
        my_counter = my_counter +1
        print ('Iter:', my_counter)
        
        
        plt.figure(figsize=(6,5),dpi=100)
        plt.plot(x,y,'r',label='true')
        plt.ylim([-0.5,0.5])
        plt.savefig('./opt_plot/%s.png'%my_counter,format='png')
        plt.close()
           
                
        return  e
    
    
    fn=foil[jj]  

         
    idx=np.argwhere(name=='%s'%fn)
    p1=mypara[idx[0][0],:]/scaler
    
    
    fp_conv=open(path+'conv_%s.dat'%fn,'w+')
    
       
    print('Starting loss = {}'.format(loss(p1)))
    print('Intial foil = %s' %name[idx[0]])
    
    mylimit=((0,1.1),(0,1.1),(0.2,1.1))
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
    
    fp=open(path+'resx_%s.dat'%fn,'w')
    fp.write('tar-cl = %s \n'%tar_cl)
    fp.write('init-cl = %s \n'%init_cl)
    fp.write('pred-cl = %s \n'%pred_cl)
    fp.write('tar-cd = %s \n'%tar_cd)
    fp.write('init-cd = %s \n'%init_cd)
    fp.write('pred-cd = %s \n'%pred_cd)   
    fp.write('tar-cl/cd = %s \n'%(tar_cl/tar_cd))
    fp.write('init-cl/cd = %s \n'%(init_cl/init_cd))
    fp.write('pred-cl/cd = %s \n'%(pred_cl/pred_cd))      
        
    fp.write('Re = %s \n'%(reno*100000))
    fp.write('aoa = %s \n'%(aoa*14))
    fp.write('%s \n'%res.x)
    fp.write('%s %s %s \n'%(res.x*[6,6,30][0],res.x*[6,6,30][1],res.x*[6,6,30][2]))
    fp.close()
        
    #intial shape
    x0,y0=get_coord(p1)
    fp=open(path+'base_%s.dat'%fn,'w')
    for i in range(len(x0)):
        fp.write('%f %f 0.00\n'%(x0[i],y0[i]))
    fp.close()
        
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(x0,y0,'--k',label='Base')
    plt.plot(x,y,'g',lw=3,label='Optimized')
    plt.legend(fontsize=14)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.25,0.25])
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)
    plt.savefig(path+'opti_%s.png'%fn,format='png',bbox_inches='tight',dpi=300)
    plt.close()
        
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(reno*100000,tar_cl,'-ok',lw=3,label='Target')
    plt.plot(reno*100000,init_cl,'-ob',lw=3,label='Base')
    plt.plot(reno*100000,pred_cl,'-og',lw=3,label='Optimized')
       
    plt.legend(fontsize=14)
    plt.xlabel('Reynolds No',fontsize=16)
    plt.ylabel('Cl',fontsize=16)
    plt.ylim([0.4,1.2])
    plt.savefig(path+'cl_%s.png'%fn,format='png',bbox_inches='tight',dpi=300)
    plt.close()
    
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(reno*100000,tar_cd,'-ok',lw=3,label='Target')
    plt.plot(reno*100000,init_cd,'-ob',lw=3,label='Base')
    plt.plot(reno*100000,pred_cd,'-og',lw=3,label='Optimized')
       
    plt.legend(fontsize=14)
    plt.xlabel('Reynolds No',fontsize=16)
    plt.ylabel('Cd',fontsize=16)
    plt.ylim([0.0,0.1])
    plt.savefig(path+'cd_%s.png'%fn,format='png',bbox_inches='tight',dpi=300)
    plt.close()
    
    
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(reno*100000,tar_cl/tar_cd,'--ok',lw=3,label='Target')
    plt.plot(reno*100000,init_cl/init_cd,'--ob',lw=3,label='Base')
    plt.plot(reno*100000,pred_cl/pred_cd,'--og',lw=3,label='Optimized')
       
    plt.legend(fontsize=14)
    plt.xlabel('Reynolds No',fontsize=16)
    plt.ylabel('Cl/Cd',fontsize=16)
    plt.ylim([0.4,30])
    plt.savefig(path+'clcd_%s.png'%fn,format='png',bbox_inches='tight',dpi=300)
    plt.close()


        
    fp_conv.close()