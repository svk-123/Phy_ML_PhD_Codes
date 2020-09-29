#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

with new parameters tanh_16_v1:

with new flow prediction network using v1.

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


# ref:[data,name]
folder = './tmp/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)



global xx
mypara=[]
name=[]
xx=[]

#load airfoil para
path='./data_file_cnn/'
data_file='cnn_uiuc_para_8_tanh_v1.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)

mypara.extend(result1[0])
name.extend(result1[1])
xx.extend(result1[2])

#myscaler=result1[7] #scaled parameter

##load airfoil para
#path='./data_file_cnn/'
#data_file='naca4_cnn_clcd_turb_8para_v1.pkl'
#with open(path + data_file, 'rb') as infile:
#    result2 = pickle.load(infile)
#
#cd.extend(result2[1])
#cl.extend(result2[2])
#mypara.extend(result2[5])
#name.extend(result2[6])


mypara=np.asarray(mypara)
name=np.asarray(name)
xx=np.asarray(xx)

#with open('xx.npy', 'wb') as f:
#    np.save(f, xx)
    
#nname=[]
#for i in range(len(name)):
#    nname.append(name[i].decode())
#name=np.asarray(nname)

#fp=open('cl_data_turb_gen_st.txt','w')
#for i in range(len(cl)):
#    fp.write('%s %f %f %f \n'%(name[i],myreno[i],myaoa[i],cl[i]))
#    
#fp.close()


model_cl=load_model('./selected_model_cnn/case_gen_naca4_80x6/model/final_sf.hdf5')  
model_para=load_model('./selected_model_cnn/case_1_p8_tanh_include_naca4_v2/model_cnn/final_cnn.hdf5') 
get_c= K.function([model_para.layers[16].input],  [model_para.layers[19].output])

global scaler
global tar_cl
global init_cl
global reno
global aoa

tar_cl=1.5
init_cl=0
reno=np.asarray([50000])/100000.
aoa=np.asarray([6])/14.

reno=np.asarray(reno)
aoa=np.asarray(aoa)

global my_counter
my_counter =0

global error
error=[]




def get_coord(p2):
    
    para1=p2
    para1=np.reshape(para1,(1,8))
    c1 = get_c([para1])[0][0,:]
    c1=c1*0.2
    c2=np.concatenate((c1[0:100],-c1[0:1]),axis=0)
    return (np.asarray([xx,c2]).transpose())

def loss(para):
       
    global my_counter  
    global init_cl
    mypara=para

    xy=get_coord(mypara)

    my_inp=np.concatenate((reno,aoa,mypara),axis=0)
    my_inp=np.reshape(my_inp,(1,10))
    #cd, cl
    out=model_cl.predict([my_inp])
    out=out*np.asarray([0.33,2.05])
                
    pred_cl=out[0,1]
    pred_cd=out[0,0]
    
    print ('Pred_cl:', pred_cl)
#    if(pred_cl > tar_cl):
#        #e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
#        e=0
#    else:
#        e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
    
    if (pred_cl > 0):
        e=(pred_cd/pred_cl)*100
    else:
        e=100.0
    print ('mse:', e)
       
    
    fp.write('%s %s %s\n'%(my_counter,e,pred_cl))    
    if(my_counter == 0):
        init_cl=pred_cl
    my_counter = my_counter +1
    print ('Iter:', my_counter)
    
     
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xy[:,0],xy[:,1],'r',label='true')
    plt.ylim([-0.5,0.5])
    plt.savefig('./tmp/%s.png'%my_counter,format='png')
    plt.close()
                  
    return  e

np.random.seed(123)     
for iters in range(10):
    
    my_counter=0
    
    #base foil name
    idx1=np.random.randint(1425)   
    fn=name[idx1]
    #fn='naca0006'
    print(fn)
    idx=np.argwhere(name=='%s'%fn)
    
    #scaled parameter
    p1=mypara[idx[0][0],:]
    
    #conv file
    fp=open('./opti_plot/conv_%s.dat'%fn,'w+')
     
    print('Starting loss = {}'.format(loss(p1)))
    print('Intial foil = %s' %name[idx[0]])
    
    a=mypara.copy()
    a1=-0.0
    a2=0.0
    mylimit=((a[:,0].min()+a1,a[:,0].max()+a2),(a[:,1].min()+a1,a[:,1].max()+a2),(a[:,2].min()+a1,a[:,2].max()+a2),\
             (a[:,3].min()+a1,a[:,3].max()+a2),(a[:,4].min()+a1,a[:,4].max()+a2),\
             (a[:,5].min()+a1,a[:,5].max()+a2),(a[:,6].min()+a1,a[:,6].max()+a2),\
             (a[:,7].min()+a1,a[:,7].max()+a2))
    
    #mylimit=((0,6),(0,6),(6,32))
    res = minimize(loss, x0=p1, method = 'L-BFGS-B', bounds=mylimit, \
                   options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
                                     'eps': 0.1, 'maxfun': 100, \
                                     'maxiter': 50, 'maxls': 100})
        
    #res = minimize(loss, x0=p1, method = 'L-BFGS-B', \
    #               options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
    #                                 'eps': 0.01, 'maxfun': 100, \
    #                                 'maxiter': 50, 'maxls': 100})
        
    print('Ending loss = {}'.format(loss(res.x)))
    fp.close()
        
    fp=open('./opti_plot/final_%s.dat'%fn,'w')
    xy=get_coord(res.x)
    for i in range(len(xy)):
        fp.write('%f %f 0.00\n'%(xy[i,0],xy[i,0]))
    fp.close()
    
    fp=open('./opti_plot/resx_%s.dat'%fn,'w')
    fp.write('%s'%res.x)
    fp.close()
    
    #intial shape
    xy0=get_coord(p1)
        
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xy0[:,0],xy0[:,1],'--k',label='Base')
    plt.plot(xy[:,0],xy[:,1],'g',lw=3,label='Optimized')
    plt.legend(fontsize=14)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.25,0.25])
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)
    plt.savefig('./opti_plot/opti_%s.png'%fn,format='png',bbox_inches='tight',dpi=300)
    plt.close()
    
    
    