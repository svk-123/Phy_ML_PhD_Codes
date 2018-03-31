#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

this is test for one layer RBF and NN together.

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
import math

#load data
xtmp=[]
ytmp=[]
reytmp=[]
utmp=[]
vtmp=[]

flist=['Re100']
for ii in range(len(flist)):
    #x,y,Re,u,v
    with open('./data/cavity_%s.pkl'%flist[ii], 'rb') as infile:
        result = pickle.load(infile)
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    reytmp.extend(result[2])
    utmp.extend(result[3])
    vtmp.extend(result[4])
    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
reytmp=np.asarray(reytmp)
utmp=np.asarray(utmp)
vtmp=np.asarray(vtmp)    

#normalize
reytmp=reytmp/1000.
val_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
val_out=np.concatenate((utmp[:,None],vtmp[:,None]),axis=1)    


#load_model
model_test=load_model('./selected_model/uv_both/final_sf.hdf5') 
out=model_test.predict([val_inp])    

#layer-1 centres
with open('./rbfcom/data_cavity_c200_l1.pkl', 'rb') as infile:
    result = pickle.load(infile)
print('found centers')
c=result[0]

d=2
sp1=0.0

#prediction-x
x=val_inp[:,0:2]
y=val_out[:,0]

from mrbf_layers import layer_1
l1u=layer_1(x,y,c,x.shape[0],c.shape[0],2,sp1)
l1u.load_weight1(0)
l1u.pred_f_ga()
predu=l1u.pred

#prediction-y
x=val_inp[:,0:2]
y=val_out[:,1]

l1v=layer_1(x,y,c,x.shape[0],c.shape[0],2,sp1)
l1v.load_weight1(1)
l1v.pred_f_ga()
predv=l1v.pred

        
#plot
def line_plot1():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(u1a,ya,'-og',linewidth=2,label='true')
    #plt0, =plt.plot(u2a,ya,'r',linewidth=2,label='nn')
    plt0, =plt.plot(ru2a,ya,'b',linewidth=2,label='rbf')
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
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(xb,v1a,'-og',linewidth=2,label='true')
    #plt0, =plt.plot(xb,v2a,'r',linewidth=2,label='nn')    
    plt0, =plt.plot(xb,rv2a,'b',linewidth=2,label='rbf')  
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
#plot(xtmp,ytmp,out[:,0],20,'u-nn')
#plot(xtmp,ytmp,abs(out[:,0]-val_out[:,0]),20,'u-nn-error')
plot(xtmp,ytmp,predu,20,'u-rbf')
#plot(xtmp,ytmp,abs(predu-val_out[:,0]),20,'u-rbf-error')
    
plot(xtmp,ytmp,val_out[:,1],20,'v-cfd')
#plot(xtmp,ytmp,out[:,1],20,'v-nn')
#plot(xtmp,ytmp,abs(out[:,1]-val_out[:,1]),20,'v-nn-error')
plot(xtmp,ytmp,predv,20,'v-rbf')
#plot(xtmp,ytmp,abs(predv-val_out[:,1]),20,'v-rbf-error')


#LinearNDinterpolator - for nn
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


#LinearNDinterpolator - for rbf
pD=np.asarray([xtmp,ytmp]).transpose()
print 'interpolation-1...'      
f1u=interpolate.LinearNDInterpolator(pD,val_out[:,0])

ru1a=np.zeros((len(ya)))
ru1b=np.zeros((len(ya)))
for i in range(len(ya)):
    ru1a[i]=f1u(xa[i],ya[i])
    ru1b[i]=f1u(xb[i],yb[i])

print 'interpolation-2...'      
f2u=interpolate.LinearNDInterpolator(pD,predu)

ru2a=np.zeros((len(ya)))
ru2b=np.zeros((len(ya)))
for i in range(len(ya)):
    ru2a[i]=f2u(xa[i],ya[i])
    ru2b[i]=f2u(xb[i],yb[i])

print 'interpolation-3...'      
f1v=interpolate.LinearNDInterpolator(pD,val_out[:,1])

rv1a=np.zeros((len(ya)))
rv1b=np.zeros((len(ya)))
for i in range(len(ya)):
    rv1a[i]=f1v(xb[i],yb[i])
    rv1b[i]=f1v(xa[i],ya[i])

print 'interpolation-4...'      
f2v=interpolate.LinearNDInterpolator(pD,predv)

rv2a=np.zeros((len(ya)))
rv2b=np.zeros((len(ya)))
for i in range(len(ya)):
    rv2a[i]=f2v(xb[i],yb[i])
    rv2b[i]=f2v(xa[i],ya[i])


line_plot1()
line_plot2()







