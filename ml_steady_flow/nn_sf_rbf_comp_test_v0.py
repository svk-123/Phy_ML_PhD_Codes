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
model_test=load_model('./selected_model/6x100/final_sf.hdf5') 
mlp_out=model_test.predict([val_inp])  
  
with open('./selected_rbf_model/case_4_4000/pred/%s.pkl'%flist[ii], 'rb') as infile:
    result = pickle.load(infile)  
rbf_out=result[0]

#plot
def line_plot1():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(u1a,ya_cfd,'g',marker='o',mew=1.5, mfc='None',lw=2,ms=12,markevery=1,label='CFD')
    plt0, =plt.plot(u2a_mlp,ya,'r',linewidth=3,label='MLP')
    plt0, =plt.plot(u2a_rbf,ya,'--b',linewidth=3,label='RBF')
    plt.legend(fontsize=20)
    plt.xlabel('u-velocity',fontsize=20)
    plt.ylabel('Y',fontsize=20)
    #plt.title('%s-u'%(flist[ii]),fontsiuze=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./comp/%s-u.png'%(flist[ii]), format='png',bbox_inches='tight', dpi=200)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(xb_cfd,v1a,'g',marker='o',mew=1.5, mfc='None',lw=2,ms=12,markevery=1,label='CFD')
    plt0, =plt.plot(xb,v2a_mlp,'r',linewidth=3,label='MLP')    
    plt0, =plt.plot(xb,v2a_rbf,'--b',linewidth=3,label='RBF')  
    plt.legend(fontsize=20)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('v-velocity' ,fontsize=20)
    #plt.title('%s-v'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./comp/%s-v.png'%(flist[ii]), format='png',bbox_inches='tight', dpi=200)
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
    plt.savefig('./comp/%s.png'%(flist[ii]+'_'+name), format='png',bbox_inches='tight', dpi=200)
    plt.show()
          
plot(xtmp,ytmp,val_out[:,0],20,'u-cfd')
plot(xtmp,ytmp,mlp_out[:,0],20,'u-mlp')
plot(xtmp,ytmp,rbf_out[:,0],20,'u-rbf')
plot(xtmp,ytmp,abs(mlp_out[:,0]-val_out[:,0]),20,'u-mlp_error')
plot(xtmp,ytmp,abs(rbf_out[:,0]-val_out[:,0]),20,'u-rbf_error')
    
plot(xtmp,ytmp,val_out[:,1],20,'v-cfd')
plot(xtmp,ytmp,mlp_out[:,1],20,'v-mlp')
plot(xtmp,ytmp,abs(mlp_out[:,1]-val_out[:,1]),20,'v-mlp_error')
plot(xtmp,ytmp,rbf_out[:,1],20,'v-mlp')
plot(xtmp,ytmp,abs(rbf_out[:,1]-val_out[:,1]),20,'v-rbf_error')

#LinearNDinterpolator
pD=np.asarray([xtmp,ytmp]).transpose()
    
print 'interpolation-1...'      
f1u=interpolate.LinearNDInterpolator(pD,val_out[:,0])
xa=np.linspace(0.5,0.5,50)
ya=np.linspace(0.01,0.99,50)
xb=ya
yb=xa

xa_cfd=np.linspace(0.5,0.5,25)
ya_cfd=np.linspace(0.01,0.99,25)
xb_cfd=ya_cfd
yb_cfd=xa_cfd
u1a=np.zeros((len(ya_cfd)))
for i in range(len(ya_cfd)):
    u1a[i]=f1u(xa[i],ya_cfd[i])

print 'interpolation-2...'      
f2u_mlp=interpolate.LinearNDInterpolator(pD,mlp_out[:,0])
u2a_mlp=np.zeros((len(ya)))
for i in range(len(ya)):
    u2a_mlp[i]=f2u_mlp(xa[i],ya[i])

print 'interpolation-2...'      
f2u_rbf=interpolate.LinearNDInterpolator(pD,rbf_out[:,0])
u2a_rbf=np.zeros((len(ya)))
for i in range(len(ya)):
    u2a_rbf[i]=f2u_rbf(xa[i],ya[i])

print 'interpolation-3...'      
f1v=interpolate.LinearNDInterpolator(pD,val_out[:,1])
v1a=np.zeros((len(ya_cfd)))
for i in range(len(ya_cfd)):
    v1a[i]=f1v(xb_cfd[i],yb_cfd[i])

print 'interpolation-4...'      
f2v_mlp=interpolate.LinearNDInterpolator(pD,mlp_out[:,1])
v2a_mlp=np.zeros((len(ya)))
for i in range(len(ya)):
    v2a_mlp[i]=f2v_mlp(xb[i],yb[i])

print 'interpolation-4...'      
f2v_rbf=interpolate.LinearNDInterpolator(pD,rbf_out[:,1])
v2a_rbf=np.zeros((len(ya)))
for i in range(len(ya)):
    v2a_rbf[i]=f2v_rbf(xb[i],yb[i])


line_plot1()
line_plot2()


tmp=val_out[:,0]-mlp_out[:,0]
mlp_l2_u=(LA.norm(tmp)/LA.norm(val_out[:,0]))*100 
tmp=val_out[:,0]-rbf_out[:,0]
rbf_l2_u=(LA.norm(tmp)/LA.norm(val_out[:,0]))*100 

tmp=val_out[:,1]-mlp_out[:,1]
mlp_l2_v=(LA.norm(tmp)/LA.norm(val_out[:,1]))*100 
tmp=val_out[:,1]-rbf_out[:,1]
rbf_l2_v=(LA.norm(tmp)/LA.norm(val_out[:,1]))*100 
