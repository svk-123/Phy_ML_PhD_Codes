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


#load airfoil para
path='./data_file/'
data_file='naca4_clcd_turb_st_8para_tanh_cnn.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)

cd.extend(result1[1])
cl.extend(result1[2])
mypara.extend(result1[5])
name.extend(result1[6])
myscaler=np.array([1.,1.,1.,1.,1.,1.,1.,1.])


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



model_cl=load_model('./selected_model_clcd/case_1_clcd_8para_cnn_80x6/model/model_sf_600_0.00001013_0.00001612.hdf5')  
#model_cl=load_model('./selected_model/turb_naca4_8para_6x30/final_sf.hdf5')  
model_para=load_model('./selected_model_cnn_para/case_1_p8_naca4_C5F7/model_cnn/final_cnn.hdf5') 
get_c= K.function([model_para.layers[16].input],  [model_para.layers[19].output])

global scaler
scaler = myscaler

global tar_cl
global init_cl
global reno
global aoa

tar_cl=1.0
init_cl=0
reno=np.asarray([50000])/100000.
aoa=np.asarray([6])/14.

reno=np.asarray(reno)
aoa=np.asarray(aoa)

global my_counter
my_counter =0

global error
error=[]

global xx
xx=np.loadtxt('./data_file/naca0006.dat',skiprows=1)
xx=xx[:100,0]

def get_coord(p2):
    
    para1=p2*scaler
    para1=np.reshape(para1,(1,8))
    c1 = get_c([para1])[0][0,:]
    c1=c1*0.2
    return (xx,c1)

def loss(para):
       
    global my_counter  
    global init_cl
    mypara=para

    x,y=get_coord(mypara)

    my_inp=np.concatenate((reno,aoa,mypara),axis=0)
    my_inp=np.reshape(my_inp,(1,10))
    #cd, cl
    out=model_cl.predict([my_inp])
    out=out*np.asarray([0.33,2.05])
                
    pred_cl=out[0,1]
    
    print ('Pred_cl:', pred_cl)
    if(pred_cl > tar_cl):
        #e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
        e=0
    else:
        e=np.sqrt(((tar_cl - pred_cl) ** 2).mean())
    print ('mse:', e)
       
    
    fp.write('%s %s %s\n'%(my_counter,e,pred_cl))    
    if(my_counter == 0):
        init_cl=pred_cl
    my_counter = my_counter +1
    print ('Iter:', my_counter)
    
     
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(x,y,'r',label='true')
    plt.ylim([-0.5,0.5])
    plt.savefig('./opt_plot/%s.png'%my_counter,format='png')
    plt.close()
                  
    return  e
     

#base foil name
fn='naca0012'

idx=np.argwhere(name=='%s'%fn)

#scaled parameter
p1=mypara[idx[0][0],:]

#conv file
fp=open('./tmp/conv_%s.dat'%fn,'w+')
 
print('Starting loss = {}'.format(loss(p1)))
print('Intial foil = %s' %name[idx[0]])

a=mypara.copy()
mylimit=((a[:,0].min(),a[:,0].max()),(a[:,1].min(),a[:,1].max()),(a[:,2].min(),a[:,2].max()),(a[:,3].min(),a[:,3].max()),\
         (a[:,4].min(),a[:,4].max()),(a[:,5].min(),a[:,5].max()),(a[:,6].min(),a[:,6].max()),\
         (a[:,7].min(),a[:,7].max()))
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


fp=open('./tmp/final_%s.dat'%fn,'w')
x,y=get_coord(res.x)
for i in range(len(x)):
    fp.write('%f %f 0.00\n'%(x[i],y[i]))
fp.close()

fp=open('./tmp/resx_%s.dat'%fn,'w')
fp.write('%s'%res.x)
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
plt.savefig('./tmp/opti_%s.png'%fn,format='png',bbox_inches='tight',dpi=300)
plt.close()