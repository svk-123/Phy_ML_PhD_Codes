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
import cPickle as pickle
import pandas

from scipy import interpolate
from numpy import linalg as LA
       

#load data
xtmp=[]
ytmp=[]
reytmp=[]
utmp=[]
vtmp=[]
ptmp=[]

flist=['Re10000']
for ii in range(len(flist)):
    #x,y,Re,u,v
    with open('./data/cavity_%s.pkl'%flist[ii], 'rb') as infile:
        result = pickle.load(infile)
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    reytmp.extend(result[2])
    utmp.extend(result[3])
    vtmp.extend(result[4])
    ptmp.extend(result[5])    
    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
reytmp=np.asarray(reytmp)
utmp=np.asarray(utmp)
vtmp=np.asarray(vtmp)    
ptmp=np.asarray(ptmp) 

#normalize
reytmp=reytmp/10000.
val_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
val_out=np.concatenate((utmp[:,None],vtmp[:,None],ptmp[:,None]),axis=1)    

#load_model
model_test=load_model('./selected_model/Re1000-10000/final_sf.hdf5') 
out=model_test.predict([val_inp])    
  
#plot
def line_plot1():
    plt.figure(figsize=(8, 5), dpi=100)
    plt0, =plt.plot(u1a,ya,'-og',linewidth=2,label='true')
    plt0, =plt.plot(u2a,ya,'r',linewidth=2,label='nn')
    plt.legend(fontsize=16)
    plt.xlabel('u',fontsize=16)
    plt.ylabel('y',fontsize=16)
    plt.title('%s-u'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('%s-u'%(flist[ii]), format='png', dpi=100)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(8, 5), dpi=100)
    plt0, =plt.plot(xb,v1a,'-og',linewidth=2,label='true')
    plt0, =plt.plot(xb,v2a,'r',linewidth=2,label='nn')    
    plt.legend(fontsize=16)
    plt.xlabel('x ',fontsize=16)
    plt.ylabel('v' ,fontsize=16)
    plt.title('%s-v'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('%s-v'%(flist[ii]), format='png', dpi=100)
    plt.show()     
    
#plot
def plot(xp,yp,zp,nc,name):

    plt.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
    v= np.linspace(0, 0.05, 15, endpoint=True)
    #cp = plt.tricontourf(xp,yp,zp,v,cmap=cm.jet,extend='both')
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    plt.colorbar()
    plt.title('%s  '%flist[ii]+name)
    plt.xlabel('X ')
    plt.ylabel('Y ')
    plt.savefig('%s'%flist[ii]+name, format='png', dpi=100)
    plt.show()
          
plot(xtmp,ytmp,val_out[:,0],20,'u-cfd')
plot(xtmp,ytmp,out[:,0],20,'u-nn')
plot(xtmp,ytmp,abs(out[:,0]-val_out[:,0]),20,'u-error')
    
plot(xtmp,ytmp,val_out[:,1],20,'v-cfd')
plot(xtmp,ytmp,out[:,1],20,'v-nn')
plot(xtmp,ytmp,abs(out[:,1]-val_out[:,1]),20,'v-error')



#LinearNDinterpolator
pD=np.asarray([xtmp,ytmp]).transpose()
    
print 'interpolation-1...'      
f1u=interpolate.LinearNDInterpolator(pD,val_out[:,0])
xa=np.linspace(0.5,0.5,50)
ya=np.linspace(0.01,0.99,50)
xb=ya
yb=xa
u1a=np.zeros((len(ya)))
u1b=np.zeros((len(ya)))
for i in range(len(ya)):
    u1a[i]=f1u(xa[i],ya[i])
    u1b[i]=f1u(xb[i],yb[i])

print 'interpolation-2...'      
f2u=interpolate.LinearNDInterpolator(pD,out[:,0])

u2a=np.zeros((len(ya)))
u2b=np.zeros((len(ya)))
for i in range(len(ya)):
    u2a[i]=f2u(xa[i],ya[i])
    u2b[i]=f2u(xb[i],yb[i])

print 'interpolation-3...'      
f1v=interpolate.LinearNDInterpolator(pD,val_out[:,1])

v1a=np.zeros((len(ya)))
v1b=np.zeros((len(ya)))
for i in range(len(ya)):
    v1a[i]=f1v(xb[i],yb[i])
    v1b[i]=f1v(xa[i],ya[i])

print 'interpolation-4...'      
f2v=interpolate.LinearNDInterpolator(pD,out[:,1])

v2a=np.zeros((len(ya)))
v2b=np.zeros((len(ya)))
for i in range(len(ya)):
    v2a[i]=f2v(xb[i],yb[i])
    v2b[i]=f2v(xa[i],ya[i])


line_plot1()
line_plot2()

func_name='nn'
data_rbf=[u1a,ya,u2a,ya,xb,v1a,xb,v2a]
with open('./plot/cavity_%s_%s.pkl'%(func_name,flist[0]), 'wb') as outfile:
    pickle.dump(data_rbf, outfile, pickle.HIGHEST_PROTOCOL)




