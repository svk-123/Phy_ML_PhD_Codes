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
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
plt.rc('font', family='serif')

#load data
xtmp=[]
ytmp=[]
reytmp=[]
utmp=[]
vtmp=[]
ptmp=[]

flist=['Re1000']
for ii in range(len(flist)):
    #x,y,Re,u,v
    with open('./data/cavity_%s_part_y.pkl'%flist[ii], 'rb') as infile:
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
reytmp=reytmp/1000.
val_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
val_out=np.concatenate((utmp[:,None],vtmp[:,None],ptmp[:,None]),axis=1)    

#load_model
model_test=load_model('./thesis_selected_model/case_1_6x100_tanh/model/final_sf.hdf5') 
out=model_test.predict([val_inp])    
  
#plot
def line_plot1():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(u1a,ya,'g',marker='o',mew=1.5, mfc='None',lw=3,ms=12,markevery=1,label='CFD')
    plt0, =plt.plot(u2a,ya,'r',linewidth=3,label='MLP')
    plt0, =plt.plot(u2a1,ya,'b',linewidth=3,label='MLP-Local')
    plt.legend(fontsize=20)
    plt.xlabel('u-velocity',fontsize=20)
    plt.ylabel('Y',fontsize=20)
    #plt.title('%s-u'%(flist[ii]),fontsiuze=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.5,0)
    #plt.ylim(-0.01,0.5)    
    plt.savefig('%s-u_part.png'%(flist[ii]), format='png',bbox_inches='tight', dpi=200)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(xb,v1a,'g',marker='o',mew=1.5, mfc='None',lw=3,ms=12,markevery=1,label='CFD')
    plt0, =plt.plot(xb,v2a,'r',linewidth=3,label='MLP')   
    plt0, =plt.plot(xb,v2a1,'b',linewidth=3,label='MLP-Local')  
    plt.legend(fontsize=20)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('v-velocity' ,fontsize=20)
    #plt.title('%s-v'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('%s-v_part.png'%(flist[ii]), format='png',bbox_inches='tight', dpi=200)
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
    #plt.title('%s  '%flist[ii]+name)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    plt.savefig('%s'%flist[ii]+name, format='png',bbox_inches='tight', dpi=100)
    plt.show()
          
'''plot(xtmp,ytmp,val_out[:,0],20,'u-cfd')
plot(xtmp,ytmp,out[:,0],20,'u-nn')
plot(xtmp,ytmp,abs(out[:,0]-val_out[:,0]),20,'u-error')
    
plot(xtmp,ytmp,val_out[:,1],20,'v-cfd')
plot(xtmp,ytmp,out[:,1],20,'v-nn')
plot(xtmp,ytmp,abs(out[:,1]-val_out[:,1]),20,'v-error')'''



#LinearNDinterpolator
pD=np.asarray([xtmp,ytmp]).transpose()


xa=np.linspace(0.5,0.5,50)
ya=np.linspace(0.01,0.4,50)
xb=ya
yb=xa
    
print 'interpolation-1...'      
f1u=interpolate.LinearNDInterpolator(pD,val_out[:,0])

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

model_test1=load_model('./thesis_selected_model/case_1_part_y_6x50_tanh/model_sf_1000_0.000001_0.000001.hdf5') 
out1=model_test1.predict([val_inp])  


print 'interpolation-2...'      
f2u1=interpolate.LinearNDInterpolator(pD,out1[:,0])

u2a1=np.zeros((len(ya)))
u2b1=np.zeros((len(ya)))
for i in range(len(ya)):
    u2a1[i]=f2u1(xa[i],ya[i])
    u2b1[i]=f2u1(xb[i],yb[i])


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

print 'interpolation-4...'      
f2v1=interpolate.LinearNDInterpolator(pD,out1[:,1])

v2a1=np.zeros((len(ya)))
v2b1=np.zeros((len(ya)))
for i in range(len(ya)):
    v2a1[i]=f2v1(xb[i],yb[i])
    v2b1[i]=f2v1(xa[i],ya[i])









line_plot1()
line_plot2()






