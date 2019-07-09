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

#load data
xtmp=[]
ytmp=[]
p=[]
u=[]
v=[]
p_pred=[]
u_pred=[]
v_pred=[]

flist=['rexxx']
for ii in range(len(flist)):
    #x,y,Re,u,v
    with open('./pred/ldc/pred_ldc_%s.pkl'%flist[ii], 'rb') as infile:
        result = pickle.load(infile)
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    p.extend(result[2])
    u.extend(result[3])
    v.extend(result[4])    
    p_pred.extend(result[5])
    u_pred.extend(result[6])
    v_pred.extend(result[7]) 
    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
p=np.asarray(p)
u=np.asarray(u)
v=np.asarray(v)
p_pred=np.asarray(p_pred)
u_pred=np.asarray(u_pred)
v_pred=np.asarray(v_pred)


#plot
def line_plot1():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(u1a,ya,'-og',linewidth=3,label='true')
    plt0, =plt.plot(u2a,ya,'r',linewidth=3,label='NN')
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
    plt.legend(fontsize=20)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('v-velocity' ,fontsize=20)
    #plt.title('%s-v'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-v'%(flist[ii]), format='png',bbox_inches='tight', dpi=100)
    plt.show()     
    
#plot
def plot(xp,yp,zp,nc,name):

    plt.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = plt.tricontour(xp,yp,zp,nc,linewidths=0.3,colors='k',zorder=5)
    cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet,zorder=0)
   # v= np.linspace(0, 0.05, 15, endpoint=True)
    #cp = plt.tricontourf(xp,yp,zp,v,cmap=cm.jet,extend='both')
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    plt.colorbar()
    #plt.title('%s  '%flist[ii]+name)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    plt.savefig('./plot/%s'%flist[ii]+name, format='png',bbox_inches='tight', dpi=100)
    plt.show()
          
plot(xtmp,ytmp,u[:,0],20,'u-cfd')
plot(xtmp,ytmp,u_pred[:,0],20,'u-nn')
plot(xtmp,ytmp,abs(u[:,0]-u_pred[:,0]),20,'u-error')
    
plot(xtmp,ytmp,v[:,0],20,'v-cfd')
plot(xtmp,ytmp,v_pred[:,0],20,'v-nn')
plot(xtmp,ytmp,abs(v[:,0]-v_pred[:,0]),20,'v-error')


#LinearNDinterpolator
pD=np.asarray([xtmp,ytmp]).transpose()
    
print ('interpolation-1...')      
f1u=interpolate.LinearNDInterpolator(pD,u[:,0])
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
f2u=interpolate.LinearNDInterpolator(pD,u_pred[:,0])

u2a=np.zeros((len(ya)))
u2b=np.zeros((len(ya)))
for i in range(len(ya)):
    u2a[i]=f2u(xa[i],ya[i])
    u2b[i]=f2u(xb[i],yb[i])

print ('interpolation-3...')      
f1v=interpolate.LinearNDInterpolator(pD,v[:,0])

v1a=np.zeros((len(ya)))
v1b=np.zeros((len(ya)))
for i in range(len(ya)):
    v1a[i]=f1v(xb[i],yb[i])
    v1b[i]=f1v(xa[i],ya[i])

print ('interpolation-4...')      
f2v=interpolate.LinearNDInterpolator(pD,v_pred[:,0])

v2a=np.zeros((len(ya)))
v2b=np.zeros((len(ya)))
for i in range(len(ya)):
    v2a[i]=f2v(xb[i],yb[i])
    v2b[i]=f2v(xa[i],ya[i])


line_plot1()
line_plot2()






