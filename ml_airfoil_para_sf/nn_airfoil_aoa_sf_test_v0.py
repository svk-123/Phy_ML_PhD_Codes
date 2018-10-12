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


#load data
with open('./data_file/foil_aoa_nn_test_ts_p16_NT.pkl', 'rb') as infile:
    result = pickle.load(infile)
inp_x=result[0]   
inp_y=result[1]   
inp_para=result[2]   
inp_aoa=result[3]   
out_p=result[4]   
out_u=result[5] 
out_v=result[6] 

co=result[7]

name=result[8]

inp_x=np.asarray(inp_x)
inp_y=np.asarray(inp_y)
inp_para=np.asarray(inp_para)
inp_aoa=np.asarray(inp_aoa)
out_p=np.asarray(out_p)
out_u=np.asarray(out_u)
out_v=np.asarray(out_v)

#plot
def con_plot(xp,yp,zp,nc,i,pname):

    plt.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
    plt.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
    #v= np.linspace(0, 0.05, 15, endpoint=True)
    #cp = plt.tricontourf(xp,yp,zp,v,cmap=cm.jet,extend='both')
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    plt.colorbar(cp)
    #plt.title('%s  '%flist[ii]+name)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    plt.savefig('./plot_c_ts/%s_%s_%s_aoa_%s.eps'%(i,name[i][0],pname,val_inp[2,0]), format='eps',bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()

def line_plot3(i):
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(xu[10:],pu1[10:],'og',linewidth=3,markevery=5,label='CFD-upper')
    plt0, =plt.plot(xl[:-10],pl1[:-10],'ob',linewidth=3,markevery=5,label='CFD-lower') 
            
    plt0, =plt.plot(xu,pu2,'r',linewidth=3,label='NN-upper')
    plt0, =plt.plot(xl,pl2,'k',linewidth=3,label='NN-lower')     
    
    plt.legend(fontsize=20)
    plt.xlabel('X',fontsize=20)
    plt.ylabel('$p/P_o$' ,fontsize=20)
    #plt.title('%s-AoA-%s-p'%(flist[ii],AoA[jj]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot_ts/%s_%s_aoa_%s-p.eps'%(i,name[i][0],val_inp[2,0]), format='eps',bbox_inches='tight', dpi=100)
    plt.show()    
    plt.close()

#plot
def line_plotu_sub(i):
    
    plt.figure(figsize=(8, 5), dpi=100)
    
    plt.subplot(1,4,1)
    plt.plot(u1a,ya,'-og',linewidth=3,label='CFD')
    plt.plot(u2a,ya,'r',linewidth=3,label='NN')
    plt.legend()
    plt.xlabel('                   u-velocity',fontsize=20)
    plt.ylabel('Y',fontsize=20)
    plt.xlim(0,1.2)
    
    plt.subplot(1,4,2)
    plt.plot(u1b,yb,'-og',linewidth=3)
    plt.plot(u2b,yb,'r',linewidth=3)
    plt.yticks([])
    plt.xlim(0,1.2)
    
    plt.subplot(1,4,3)
    plt.plot(u1c,yc,'-og',linewidth=3)
    plt.plot(u2c,yc,'r',linewidth=3)
    plt.yticks([])
    plt.xlim(0,1.2)
    
    plt.subplot(1,4,4)
    plt.plot(u1d,yd,'-og',linewidth=3)
    plt.plot(u2d,yd,'r',linewidth=3)
    plt.yticks([])    
    plt.xlim(0,1.2)
    
    plt.subplots_adjust(top = 1, bottom = 0.13, right = 1, left = 0.12, hspace = 0, wspace = 0)
    plt.savefig('./plot_ts/%s_%s_aoa_%s-u.eps'%(i,name[i][0],val_inp[2,0]), format='eps', dpi=100)
    plt.show()   
    plt.close()
    
#plot
def line_plotv_sub(i):
    
    fig=plt.figure(figsize=(8, 5), dpi=100)
    
    plt.subplot(1,4,1)
    plt.plot(v1a,ya,'-og',linewidth=3,label='CFD')
    plt.plot(v2a,ya,'r',linewidth=3,label='NN')
    plt.legend()
    plt.xlabel(' v-velocity',fontsize=20)
    plt.ylabel('Y',fontsize=20)
    plt.xlim(-0.1,0.5)
    
    plt.subplot(1,4,2)
    plt.plot(v1b,yb,'-og',linewidth=3)
    plt.plot(v2b,yb,'r',linewidth=3)
    plt.yticks([])
    plt.xlim(-0.1,0.25)

    plt.subplot(1,4,3)
    plt.plot(v1c,yc,'-og',linewidth=3)
    plt.plot(v2c,yc,'r',linewidth=3)
    plt.yticks([])
    plt.xlim(-0.1,0.2)
    
    plt.subplot(1,4,4)
    plt.plot(v1d,yd,'-og',linewidth=3)
    plt.plot(v2d,yd,'r',linewidth=3)
    plt.yticks([])    
    plt.xlim(-0.1,0.2)
       
    
    
    plt.subplots_adjust(top = 1, bottom = 0.13, right = 1, left = 0.12, hspace = 0, wspace = 0)
    plt.savefig(fig)
    plt.show() 
    plt.close()

for i in range(49):
    
    inp_aoa[i]=inp_aoa[i]/12.0
    val_inp=np.concatenate((inp_x[i][:,None],inp_y[i][:,None],inp_aoa[i][:,None],inp_para[i][:,:]),axis=1)
    val_out=np.concatenate((out_p[i][:,None],out_u[i][:,None],out_v[i][:,None]),axis=1)

    #load_model
    model_test=load_model('./selected_model/p16/model_sf_1400_0.000009_0.000009.hdf5') 
    out=model_test.predict([val_inp]) 
         
    con_plot(val_inp[:,0],val_inp[:,1],val_out[:,0],20,i,'p-cfd')
    con_plot(val_inp[:,0],val_inp[:,1],out[:,0],20,i,'p-nn')
    con_plot(val_inp[:,0],val_inp[:,1],abs(out[:,0]-val_out[:,0]),20,i,'p-error')
    con_plot(val_inp[:,0],val_inp[:,1],val_out[:,1],20,i,'u-cfd')
    con_plot(val_inp[:,0],val_inp[:,1],out[:,1],20,i,'u-nn')
    con_plot(val_inp[:,0],val_inp[:,1],abs(out[:,1]-val_out[:,1]),20,i,'u-error')
    con_plot(val_inp[:,0],val_inp[:,1],val_out[:,2],20,i,'v-cfd')
    con_plot(val_inp[:,0],val_inp[:,1],out[:,2],20,i,'v-nn')
    con_plot(val_inp[:,0],val_inp[:,1],abs(out[:,2]-val_out[:,2]),20,i,'v-error')
    
    #LinearNDinterpolator
    pD=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()

    xu=co[i][:100,0]
    yu=co[i][:100,1]

    xl=co[i][99:,0]
    yl=co[i][99:,1]
    
    #for -p
    print 'interpolation-1...'      
    f1p=interpolate.LinearNDInterpolator(pD,val_out[:,0])
        
    pu1=np.zeros(len(xu))
    for j in range(len(xu)):
        pu1[j]=f1p(xu[j],yu[j])
    pl1=np.zeros(len(xl))
    for j in range(len(xl)):
        pl1[j]=f1p(xl[j],yl[j])
        
    print 'interpolation-2...'      
    f2p=interpolate.LinearNDInterpolator(pD,out[:,0])
      
    pu2=np.zeros(len(xu))
    for j in range(len(xu)):
        pu2[j]=f2p(xu[j],yu[j])
    pl2=np.zeros(len(xl))
    for j in range(len(xl)):
       pl2[j]=f2p(xl[j],yl[j])

    #plot -u,v
    xa=np.linspace(co[i][99,0],co[i][99,0],50)
    ya=np.linspace(co[i][99,1],0.99,50)

    xb=np.linspace(co[i][49,0],co[i][49,0],50)
    yb=np.linspace(co[i][49,1],0.99,50)

    xc=np.linspace(co[i][0,0],co[i][0,0],50)
    yc=np.linspace(co[i][0,1],0.99,50)

    xd=np.linspace(1.5,1.5,50)
    yd=np.linspace(co[i][0,1],0.99,50)
        
    # for u    
    print 'interpolation-1...'      
    f1u=interpolate.LinearNDInterpolator(pD,val_out[:,1])
        
    u1a=np.zeros((len(ya)))
    u1b=np.zeros((len(ya)))
    u1c=np.zeros((len(ya)))
    u1d=np.zeros((len(ya)))
    for j in range(len(ya)):
        u1a[j]=f1u(xa[j],ya[j])
        u1b[j]=f1u(xb[j],yb[j])
        u1c[j]=f1u(xc[j],yc[j])
        u1d[j]=f1u(xd[j],yd[j])
        
    print 'interpolation-2...'      
    f2u=interpolate.LinearNDInterpolator(pD,out[:,1])
        
    u2a=np.zeros((len(ya)))
    u2b=np.zeros((len(ya)))
    u2c=np.zeros((len(ya)))
    u2d=np.zeros((len(ya)))
    for j in range(len(ya)):
        u2a[j]=f2u(xa[j],ya[j])
        u2b[j]=f2u(xb[j],yb[j])
        u2c[j]=f2u(xc[j],yc[j])
        u2d[j]=f2u(xd[j],yd[j])

    #for -v
    print 'interpolation-1...'      
    f1v=interpolate.LinearNDInterpolator(pD,val_out[:,2])

    v1a=np.zeros((len(ya)))
    v1b=np.zeros((len(ya)))
    v1c=np.zeros((len(ya)))
    v1d=np.zeros((len(ya)))
    for j in range(len(ya)):
        v1a[j]=f1v(xa[j],ya[j])
        v1b[j]=f1v(xb[j],yb[j])
        v1c[j]=f1v(xc[j],yc[j])
        v1d[j]=f1v(xd[j],yd[j])
   
    print 'interpolation-2...'      
    f2v=interpolate.LinearNDInterpolator(pD,out[:,2])
   
    v2a=np.zeros((len(ya)))
    v2b=np.zeros((len(ya)))
    v2c=np.zeros((len(ya)))
    v2d=np.zeros((len(ya)))
    for j in range(len(ya)):
        v2a[j]=f2v(xa[j],ya[j])
        v2b[j]=f2v(xb[j],yb[j])
        v2c[j]=f2v(xc[j],yc[j])
        v2d[j]=f2v(xd[j],yd[j])

    #plot
    line_plot3(i)
    #line_plotu_sub(i)
    #line_plotv_sub(i)

