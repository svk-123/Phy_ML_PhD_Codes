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
folder = './opti_plot/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

#folder = './tmp/'
#for the_file in os.listdir(folder):
#    file_path = os.path.join(folder, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
#    except Exception as e:
#        print(e)
        
global xx
mypara=[]
name=[]


#load airfoil para
path='./data_file_gan/'
data_file='gan_uiuc_para8_tanh_v1.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)

mypara.extend(result1[0])
name.extend(result1[1])


mypara=np.asarray(mypara)
name=np.asarray(name)
with open('./data_file_gan/xx.npy','rb') as f:
    xx=np.load(f)
xx=xx[:100]

#nname=[]
#for i in range(len(name)):
#    nname.append(name[i].decode())
#name=np.asarray(nname)

#fp=open('cl_data_turb_gen_st.txt','w')
#for i in range(len(cl)):
#    fp.write('%s %f %f %f \n'%(name[i],myreno[i],myaoa[i],cl[i]))
#    
#fp.close()

model_cl=load_model('./selected_model_gan/case_gen_gan_para_naca4_80x6/model/final_sf.hdf5')  

path='./selected_model_gan/case_1_include_naca4_5166/saved_model/'
iters=80000

# load json and create model
json_file = open(path+'aae_decoder_%s_%s.json'%(iters,iters), 'r')
loaded_model_json = json_file.read()
json_file.close()
decoder = model_from_json(loaded_model_json)
# load weights into new model
decoder.load_weights(path+"aae_decoder_%s_weights_%s.hdf5"%(iters,iters))
print("Loaded model from disk")  

# load json and create model
json_file = open(path+'aae_encoder_%s_%s.json'%(iters,iters), 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_model_json)
# load weights into new model
encoder.load_weights(path+"aae_encoder_%s_weights_%s.hdf5"%(iters,iters))
print("Loaded model from disk")  

 # load json and create model
json_file = open(path+'aae_discriminator_%s_%s.json'%(iters,iters), 'r')
loaded_model_json = json_file.read()
json_file.close()
discriminator = model_from_json(loaded_model_json)
# load weights into new model
discriminator.load_weights(path+"aae_discriminator_%s_weights_%s.hdf5"%(iters,iters))
print("Loaded model from disk")  

global scaler

global tar_cl
global init_cl
global reno
global aoa

tar_cl=1.2
init_cl=0
reno=np.asarray([20000])/100000.
aoa=np.asarray([4])/14.

reno=np.asarray(reno)
aoa=np.asarray(aoa)

global my_counter
my_counter =0

global error
error=[]


def loss(para):
       
    global init_cl
    global coord
    global my_counter
    #rescale
    mypara=para
    
    mypara_1=np.reshape(mypara,(1,8))
    
    score= discriminator.predict(mypara_1)[0][0]
    #print (score)
    coord= decoder.predict(mypara_1)
    


    my_inp=np.concatenate((reno,aoa,mypara),axis=0)
    my_inp=np.reshape(my_inp,(1,10))
        
    #cd, cl
    out=model_cl.predict([my_inp])
    out=out*np.asarray([0.33,2.04])

    pred_cl=out[0,1]
    pred_cd=out[0,0]            
    coord= decoder.predict(mypara_1)
        
    print ('Pred_cl: %0.6f %0.6f'%(pred_cl,pred_cd))
    
    #####for target cl--------------------------------
    if(pred_cl > tar_cl ):
        
        e1=0
        e2=max((0.9999-score),0)*10.0
        e3=max((coord[0,-8]+0.001)-coord[0,8],0)*100.0
        e  = e1 + e2 + e3
        
    else:
        
        e1 = np.sqrt(((tar_cl - pred_cl) ** 2).mean())
        #e1= abs(pred_cd/pred_cl)*10
        e2 = max((0.9999-score),0)*10.0
        #e3=0
        e3=max((coord[0,-8]+0.001)-coord[0,8],0)*100.0
        e  = e1 + e2 + e3
        
        
    ##### max. cl/cd------------------------------------
    '''e1= abs(pred_cd/pred_cl)*10
    e2 = max((0.9999-score),0)*10.0
    #e3=0
    e3=max((coord[0,-8]+0.001)-coord[0,8],0)*100.0
    e  = e1 + e2 + e3'''
            
    print ('mse: %0.6f %0.6f %0.6f %0.6f %0.6f'%(e, e1, e2, e3, score))
       
    fp.write('%s %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f\n'%(my_counter,pred_cl, e, e1, e2, e3, score)) 
    
    if(my_counter == 0):
        init_cl=pred_cl
    my_counter = my_counter +1
    print ('Iter:', my_counter)
    
  
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx,coord[0,:]*0.2,'r',label='true')
    plt.ylim([-0.5,0.5])
    plt.savefig('./opti_plot/%s.png'%my_counter,format='png')
    plt.close()
                  
    return  e


dest='tmp'
np.random.seed(12543)    
for iters in range(10):
    my_counter=0
    #base foil name
    idx1=np.random.randint(len(mypara))   
    fn=name[idx1]
    #fn='n0012'
    print(fn)
    idx=np.argwhere(name=='%s'%fn)
    
    #scaled parameter
    p1=mypara[idx[0][0],:]
    
    #conv file
    fp=open('./%s/conv_%s.dat'%(dest,fn),'w+')
     
    print('Starting loss = {}'.format(loss(p1)))
    print('Intial foil = %s' %name[idx[0]])
    a=mypara.copy()
    a1=-0.1
    a2=0.1
    mylimit=((a[:,0].min()+a1,a[:,0].max()+a2),(a[:,1].min()+a1,a[:,1].max()+a2),(a[:,2].min()+a1,a[:,2].max()+a2),\
             (a[:,3].min()+a1,a[:,3].max()+a2),(a[:,4].min()+a1,a[:,4].max()+a2),\
             (a[:,5].min()+a1,a[:,5].max()+a2),(a[:,6].min()+a1,a[:,6].max()+a2),\
             (a[:,7].min()+a1,a[:,7].max()+a2))
    
    #mylimit=((0,6),(0,6),(6,32))
    res = minimize(loss, x0=p1, method = 'L-BFGS-B', bounds=mylimit, \
                   options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
                                     'eps': 0.001, 'maxfun': 100, \
                                     'maxiter': 100, 'maxls': 100})
        
    #res = minimize(loss, x0=p1, method = 'L-BFGS-B', \
    #               options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
    #                                 'eps': 0.01, 'maxfun': 100, \
    #                                 'maxiter': 50, 'maxls': 100})
        
    
    print('Ending loss = {}'.format(loss(res.x)))
    fp.close()
    
    res_x=np.reshape(res.x,(1,8))
    
    fp=open('./%s/final_%s.dat'%(dest,fn),'w')
    y=decoder.predict(res_x)[0,:]
    for i in range(len(xx)):
        fp.write('%f %f 0.00\n'%(xx[i],y[i]))
    fp.close()
    
    fp=open('./%s/resx_%s.dat'%(dest,fn),'w')
    fp.write('%s'%res.x)
    fp.close()
    
    #intial shape
    init_x=np.reshape(p1,(1,8))
    y0=decoder.predict(init_x)[0,:]
    
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx,y0*0.25,'--k',label='Base')
    plt.plot(xx,y*0.25,'g',lw=3,label='Optimized')
    plt.legend(fontsize=14)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.5,0.5])
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)
    plt.savefig('./%s/opti_%s.png'%(dest,fn),format='png',bbox_inches='tight',dpi=300)
    plt.close()
    