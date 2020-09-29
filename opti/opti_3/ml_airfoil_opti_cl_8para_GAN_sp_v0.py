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
folder = './opt_plot/'
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
myreno=[]
myaoa=[]

#load airfoil para
path='./data_file/'
data_file='gen_gan_clcd_turb_RS_0p02_0p001_8para.pkl'
with open(path + data_file, 'rb') as infile:
    result2 = pickle.load(infile)

cd.extend(result2[1])
cl.extend(result2[2])
myreno.extend(result2[3])
myaoa.extend(result2[4])
mypara.extend(result2[5])
name.extend(result2[6])
myscaler=result2[7] #scaled parameter

mypara=np.asarray(mypara)
name=np.asarray(name)

#nname=[]
#for i in range(len(name)):
#    nname.append(name[i].decode())
#name=np.asarray(nname)

#fp=open('cl_data_turb_gen_st.txt','w')
#for i in range(len(cl)):
#    fp.write('%s %f %f %f \n'%(name[i],myreno[i],myaoa[i],cl[i]))
#    
#fp.close()

model_cl=load_model('./model_gan/case_1_gan_tanh_RS_0p02_0p001_clcd/model/model_sf_1600_0.00011643_0.00013853.hdf5')  

path='./model_gan/case_2_aae_uiuc_RS/saved_model/'
iters=99999

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
scaler = myscaler

global tar_cl
global init_cl
global reno
global aoa

tar_cl=1.5
init_cl=0
reno=np.asarray([50000])/100000.
aoa=np.asarray([4])/14.

reno=np.asarray(reno)
aoa=np.asarray(aoa)

global my_counter
my_counter =0

global error
error=[]

global xx
xx=np.loadtxt('./data_file/xx_101.dat')
xx=xx[:100,0]

def loss(para):
       
    global my_counter  
    global init_cl
    
    #rescale
    mypara=para
    
    mypara_1=np.reshape(mypara*scaler,(1,8))
    
    score= discriminator.predict(mypara_1)[0][0]
    #print (score)
    coord= decoder.predict(mypara_1)
    
    if(score > 0.99999):

        my_inp=np.concatenate((reno,aoa,mypara),axis=0)
        my_inp=np.reshape(my_inp,(1,10))
        
        #cd, cl
        out=model_cl.predict([my_inp])
        out=out*np.asarray([0.33,2.04])
                    
        pred_cl=out[0,1]
            
        coord= decoder.predict(mypara_1)
        
        print ('Pred_cl:', pred_cl)
        if(pred_cl > tar_cl):
            #e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
            e=0
        else:
            e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
        print ('mse:', e)
        print ('score:', score)        
        fp.write('%s %s %s\n'%(my_counter,e,pred_cl))    
        if(my_counter == 0):
            init_cl=pred_cl
        my_counter = my_counter +1
        print ('Iter:', my_counter)
        
    else:
        
        print ('skipping')   
        e=1.0
     
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx,coord[0,:]*0.2,'r',label='true')
    plt.ylim([-0.5,0.5])
    plt.savefig('./opt_plot/%s.png'%my_counter,format='png')
    plt.close()
                  
    return  e
     
for iters in range(100):
    #base foil name
    idx1=np.random.randint(80000)   
    fn=name[idx1]
    print(fn)
    idx=np.argwhere(name=='%s'%fn)
    
    #scaled parameter
    p1=mypara[idx[0][0],:]
    
    #conv file
    fp=open('./tmp/conv_%s.dat'%fn,'w+')
     
    print('Starting loss = {}'.format(loss(p1)))
    print('Intial foil = %s' %name[idx[0]])
    a=1.0
    mylimit=((0,a),(0,a),(0,a),(0,a),(0,a),(0,a),(0,a),(0,a))
    #mylimit=((0,6),(0,6),(6,32))
    res = minimize(loss, x0=p1, method = 'L-BFGS-B', bounds=mylimit, \
                   options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
                                     'eps': 0.001, 'maxfun': 100, \
                                     'maxiter': 50, 'maxls': 100})
        
    #res = minimize(loss, x0=p1, method = 'L-BFGS-B', \
    #               options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
    #                                 'eps': 0.01, 'maxfun': 100, \
    #                                 'maxiter': 50, 'maxls': 100})
        
    
    print('Ending loss = {}'.format(loss(res.x)))
    fp.close()
    
    res_x=np.reshape(res.x*scaler,(1,8))
    
    fp=open('./tmp/final_%s.dat'%fn,'w')
    y=decoder.predict(res_x)[0,:]
    #for i in range(len(x)):
    #    fp.write('%f %f 0.00\n'%(x[i],y[i]))
    #fp.close()
    
    #fp=open('./tmp/resx_%s.dat'%fn,'w')
    #fp.write('%s'%res.x)
    #fp.close()
    
    #intial shape
    init_x=np.reshape(p1*scaler,(1,8))
    y0=decoder.predict(init_x)[0,:]
    
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xx,y0*0.25,'--k',label='Base')
    plt.plot(xx,y*0.25,'g',lw=3,label='Optimized')
    plt.legend(fontsize=14)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.5,0.5])
    plt.xlabel('X',fontsize=16)
    plt.ylabel('Y',fontsize=16)
    plt.savefig('./tmp/opti_%s.png'%fn,format='png',bbox_inches='tight',dpi=300)
    plt.close()