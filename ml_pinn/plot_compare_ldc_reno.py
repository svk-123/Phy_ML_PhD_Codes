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
import matplotlib.pyplot as plt
from matplotlib import cm
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
import  pickle
import pandas

from scipy import interpolate
from numpy import linalg as LA
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 








def kextract(xtmp,ytmp,u,u_pred,v,v_pred):
    
    #LinearNDinterpolator
    pD=np.asarray([xtmp,ytmp]).transpose()
        
    print ('interpolation-1...')      
    f1u=interpolate.LinearNDInterpolator(pD,u)
    xa=np.linspace(0.5,0.5,50)
    ya=np.linspace(0.01,0.99,50)
    xb=ya
    yb=xa
    u1a=np.zeros((len(ya)))
    u1b=np.zeros((len(ya)))
    for i in range(len(ya)):
        u1a[i]=f1u(xa[i],ya[i])
        u1b[i]=f1u(xb[i],yb[i])
    
    print ('interpolation-2...')      
    f2u=interpolate.LinearNDInterpolator(pD,u_pred)
    
    u2a=np.zeros((len(ya)))
    u2b=np.zeros((len(ya)))
    for i in range(len(ya)):
        u2a[i]=f2u(xa[i],ya[i])
        u2b[i]=f2u(xb[i],yb[i])
    
    print ('interpolation-3...')      
    f1v=interpolate.LinearNDInterpolator(pD,v)
    
    v1a=np.zeros((len(ya)))
    v1b=np.zeros((len(ya)))
    for i in range(len(ya)):
        v1a[i]=f1v(xb[i],yb[i])
        v1b[i]=f1v(xa[i],ya[i])
    
    print ('interpolation-4...')      
    f2v=interpolate.LinearNDInterpolator(pD,v_pred)
    
    v2a=np.zeros((len(ya)))
    v2b=np.zeros((len(ya)))
    for i in range(len(ya)):
        v2a[i]=f2v(xb[i],yb[i])
        v2b[i]=f2v(xa[i],ya[i])
    
    return(ya,u1a,u2a,xb,v1a,v2a)





#re_train=[100,200,400,600,800,900]
    
flist_idx=np.asarray(['100','200','300','400','500','600','700','800','900','1000','1200','1500','2000'])
#flist=['100','200','300','400','500','600','700','800','900','1000','1200','1500','2000']
flist=['100']
for ii in range(len(flist)):
    
    xtmp=[]
    ytmp=[]
    p=[]
    u=[]
    v=[]
    p_pred=[]
    u_pred=[]
    v_pred=[]
    reytmp=[]
       
    #x,y,Re,u,v
    with open('./data_file_ldc/cavity_Re%s.pkl'%flist[ii], 'rb') as infile:
        result = pickle.load(infile,encoding='bytes')
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    reytmp.extend(result[2])
    u.extend(result[3])
    v.extend(result[4])
    p.extend(result[5])   
        
    xtmp=np.asarray(xtmp)
    ytmp=np.asarray(ytmp)
    u=np.asarray(u)
    v=np.asarray(v)
    p=np.asarray(p) 
    reytmp=np.asarray(reytmp)/1000  

    val_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
    val_out=np.concatenate((u[:,None],v[:,None],p[:,None]),axis=1)    

    #load_model
    model_test=load_model('./keras_model/final_sf.hdf5') 
    kout=model_test.predict([val_inp])  
    
    #pinn already generated results   
    with open('pred_ldc_re_100_2000.pkl', 'rb') as infile:
        result1 = pickle.load(infile,encoding='bytes')
    X=result1[0]
    Y=result1[1]    
    Re=result1[2]    
    U=result1[3]    
    V=result1[4]    
    P=result1[5]    
    
idx=np.argwhere(flist==flist_idx)[0][0]
ya,u1a,u2a,xb,v1a,v2a = kextract(xtmp,ytmp,u,kout[:,0],v,kout[:,1])
pya,pu1a,pu2a,pxb,pv1a,pv2a = kextract(X[idx],Y[idx],u,U[idx],v,V[idx])


#plot
def line_plot1():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(u1a,ya,'-og',linewidth=3,label='true')
    plt0, =plt.plot(u2a,ya,'r',linewidth=3,label='NN')
    plt0, =plt.plot(pu2a,pya,'b',linewidth=3,label='PINN')
    plt.legend(fontsize=20)
    plt.xlabel('u-velocity',fontsize=20)
    plt.ylabel('Y',fontsize=20)
    #plt.title('%s-u'%(flist[ii]),fontsiuze=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-u'%(flist[ii]), format='png',bbox_inches='tight', dpi=100)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(xb,v1a,'-og',linewidth=3,label='true')
    plt0, =plt.plot(xb,v2a,'r',linewidth=3,label='NN')    
    plt0, =plt.plot(pxb,pv2a,'b',linewidth=3,label='PINN')
    plt.legend(fontsize=20)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('v-velocity' ,fontsize=20)
    #plt.title('%s-v'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-v'%(flist[ii]), format='png',bbox_inches='tight', dpi=100)
    plt.show()     
        
line_plot1()
line_plot2()





