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

import matplotlib
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

## ref:[data,name]
## ref:[data,name]
#path='./airfoil_data/'
#data_file='foil_aoa_inout.pkl'
#
#with open(path + data_file, 'rb') as infile:
#    result = pickle.load(infile)
#print result[-1:]    
#
#st=800
#end=850
#
#my_inp=result[0][st:end]
#my_out=result[1][st:end]
#myco=result[4][st:end]
#mybor=result[5][st:end]
#myins=result[6][st:end]
#name=result[7][st:end]
#
#my_inp=np.asarray(my_inp)
#my_out=np.asarray(my_out)
#
#xtr1=np.reshape(my_inp,(len(my_inp),288,216,1))  
#ttr1=my_out  
#
#model_test=load_model('./selected_model/p/model_cnn_425_0.000176_0.000174.hdf5')  
##model_test=load_model('./selected_model/u/model_cnn_425_0.000293_0.000285.hdf5') 
##model_test=load_model('./selected_model/v/model_cnn_425_0.000104_0.000099.hdf5') 
#out=model_test.predict([xtr1])

def plot(zp1,zp2,zp3,nc,name):
    xp, yp = np.meshgrid(np.linspace(-1,2,216), np.linspace(1,-1,288))
    #xp, yp = np.meshgrid(np.linspace(0,216,216), np.linspace(0,288,288))
    plt.figure(figsize=(20, 5))
    
    plt.subplot(131)
    plt.contourf(xp,yp,zp1,nc,cmap=cm.jet)
    plt.colorbar()
    plt.title('CFD-%s'%name)
    
    plt.subplot(132)
    plt.contourf(xp,yp,zp2,nc,cmap=cm.jet)
    plt.colorbar()
    plt.title('Prediction-%s'%name) 
 
    plt.subplot(133)
    plt.contourf(xp,yp,zp3,nc,cmap=cm.jet)
    plt.colorbar()
    plt.title('Prediction-%s'%name) 
    
    #plt.xlim([-0.1,1.1])
    #plt.ylim([-0.1,0.1])
    plt.savefig('./plotc/p_%s.png'%(name), format='png',dpi=100)
    plt.show()


def line_plot2(l1a,l2a,l3a,lt1,lt2,lt3,lp1,lp2,lp3,name):
    plt.figure(figsize=(8, 5))
    
    lt1=lt1/2
    lt2=lt2/2
    lt3=lt3/2
    
    lp1=lp1/2
    lp2=lp2/2
    lp3=lp3/2
    
    plt0, =plt.plot(lt1+0.0,l1a,'-og',linewidth=3,label='CFD')
    plt0, =plt.plot(lt2+0.5,l2a,'-og',linewidth=3) 
    plt0, =plt.plot(lt3+1.0,l3a,'-og',linewidth=3)

    plt0, =plt.plot(lp1+0.0,l1a,'r',linewidth=3,label='CNN')
    plt0, =plt.plot(lp2+0.5,l2a,'r',linewidth=3) 
    plt0, =plt.plot(lp3+1.0,l3a,'r',linewidth=3)
    
    #plt.legend(fontsize=20)
    plt.xlabel('p-vel at x=0,0.5,1.0',fontsize=20)
    plt.ylabel('Y' ,fontsize=20)
    #plt.title('%s-AoA-%s-v'%(flist[ii],AoA[jj]),fontsize=16)
    plt.legend(loc='upper center',fontsize=20,  bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    plt.ylim(-0.05,1.3)    
    plt.savefig('./plot/u_%s.png'%(name), format='png',bbox_inches='tight', dpi=100)
    plt.show()  


def line_plot3(pt_u,pt_l,pp_u,pp_l,name):
    plt.figure(figsize=(8, 5))
        
    plt0, =plt.plot(np.linspace(0,1,len(pt_u)), pt_u,'og',linewidth=3,label='CFD')
    plt0, =plt.plot(np.linspace(0,1,len(pt_l)), pt_l,'og',linewidth=3)
    
    plt0, =plt.plot(np.linspace(0,1,len(pp_u)), pp_u,'r',linewidth=2,label='CNN')
    plt0, =plt.plot(np.linspace(0,1,len(pp_l)), pp_l,'r',linewidth=2)

    
    #plt.legend(fontsize=20)
    plt.xlabel('p - dist. ',fontsize=20)
    plt.ylabel('Y' ,fontsize=20)
    #plt.title('%s-AoA-%s-v'%(flist[ii],AoA[jj]),fontsize=16)
    plt.legend(loc='upper center',fontsize=20,  bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    plt.ylim(-0.12,-0.02)    
    plt.savefig('./plot/p_%s.png'%(name), format='png',bbox_inches='tight', dpi=100)
    plt.show() 
    

for k in range(50):
        
    p1=my_out[k].copy()
    p2=out[k,:,:,0].copy()
    
    xb=mybor[k][:,0]
    yb=mybor[k][:,1]
    
    xi=np.asarray(myins[k])[:,0]
    yi=np.asarray(myins[k])[:,1]    

    p1[xb,yb]= np.nan
    p1[yi,xi]= np.nan
    
    p2[xb,yb]= np.nan
    p2[yi,xi]= np.nan
        
    #plot(p1,p2,abs(p1-p2),20,'ts_%s'%name[k])

    y1=yb.min()
    y1_ind=np.argwhere(yb==y1)
    x1=xb[y1_ind].min()

    y2=108
    y2_ind=np.argwhere(yb==y2)
    x2=xb[y2_ind].min()
    
    y3=yb.max()
    y3_ind=np.argwhere(yb==y3)
    x3=xb[y3_ind].min()


    lt1=p1[:x1,y1][::-1]
    lt2=p1[:x2,y2][::-1]
    lt3=p1[:x3,y3][::-1]

    lp1=p2[:x1,y1][::-1]
    lp2=p2[:x2,y2][::-1]
    lp3=p2[:x3,y3][::-1]    
       
    l1a=np.asarray(range(0,x1))*0.0070422
    l2a=np.asarray(range(0,x2))*0.0070422    
    l3a=np.asarray(range(0,x3))*0.0070422   
     
    #p1[:x1,y1]=2
    #p1[:x2,y2]=2
    #p1[:x3,y3]=2
    #plot(p1,p2,abs(p1-p2),20,'ts_%s'%name[k])
    
    #line_plot2(l1a,l2a,l3a,lt1,lt2,lt3,lp1,lp2,lp3,'ts_%s'%name[k])

    # to get pressure dist
    tmp=np.concatenate((yb[:,None],xb[:,None]),axis=1)
    tmp=tmp[tmp[:,1].argsort()]
    tmp=tmp[tmp[:,0].argsort(kind='mergesort')]
    
    unique, counts = np.unique(tmp[:,0], return_counts=True)
    
    up_indx=[]
    up_indy=[]
    
    lr_indx=[]
    lr_indy=[]
    
    for jj in range(len(unique)):
        ind2=np.argwhere(tmp[:,0]==unique[jj])
        up_indx.append(unique[jj])
        up_indy.append(tmp[ind2.min(),1])
        
        lr_indx.append(unique[jj])
        lr_indy.append(tmp[ind2.max(),1])

    for kk in range(len(up_indy)):
        up_indy[kk]=up_indy[kk]
        
    for kk in range(len(lr_indy)):
        lr_indy[kk]=lr_indy[kk]  
        
    pt_u=p1[up_indx,up_indy]
    pt_l=p1[lr_indx,lr_indy]

    pp_u=p2[up_indx,up_indy]
    pp_l=p2[lr_indx,lr_indy]


    line_plot3(pt_u,pt_l,pp_u,pp_l,'ts_%s'%name[k])

