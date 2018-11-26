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
inp_x=[]
inp_y=[]
inp_reno=[]
inp_aoa=[]
inp_para=[]

out_p=[]
out_u=[]
out_v=[]

#load data
#with open('./data_file/ph_1_test/foil_aoa_nn_p16_ph_1_ts_1.pkl', 'rb') as infile:
with open('./data_file/foil_aoa_nn_naca_lam_ts_1.pkl', 'rb') as infile:
    result = pickle.load(infile)

inp_x.extend(result[0])   
inp_y.extend(result[1])
inp_para.extend(result[2])
inp_reno.extend(result[3])
inp_aoa.extend(result[4])

out_p.extend(result[5])
out_u.extend(result[6])
out_v.extend(result[7])

co=result[8]
fxy=result[9]
name=result[9]

inp_x=np.asarray(inp_x)
inp_y=np.asarray(inp_y)
inp_reno=np.asarray(inp_reno)
inp_aoa=np.asarray(inp_aoa)
inp_para=np.asarray(inp_para)

out_p=np.asarray(out_p)
out_u=np.asarray(out_u)
out_v=np.asarray(out_v)

#plot
def con_plot(xp,yp,zp,nc,i,pname):

    plt.figure(figsize=(8, 4), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
    plt.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
    #v= np.linspace(0, 0.05, 15, endpoint=True)
    #cp = plt.tricontourf(xp,yp,zp,v,cmap=cm.jet,extend='both')
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    plt.colorbar(cp)
    plt.xlim(-0.5,2)
    plt.ylim(-0.5,0.5)
    
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    plt.subplots_adjust(top = 0.95, bottom = 0.15, right = 0.98, left = 0.14, hspace = 0, wspace = 0)
    plt.savefig('./plot_ts/%s_%s_Re=%s_AoA=%s.pdf'%(pname,name[i][0],int(inp_reno[i][0]*2000),int(inp_aoa[i][0]*14)),format='pdf',dpi=200)
    plt.show()
    plt.close()

def line_plot3(i):
    
    
    mei=int(len(co[i])/30)
    if (mei < 1):
        mei=1
        
    plt.figure(figsize=(6, 4), dpi=100)
    plt0, =plt.plot(xu,pu1*2,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt0, =plt.plot(xl,pl1*2,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei) 
            
    plt0, =plt.plot(xu,pu2*2,'r',linewidth=3,label='NN')
    plt0, =plt.plot(xl,pl2*2,'r',linewidth=3)     
    
    plt.legend(fontsize=20)
    plt.xlabel('X/c',fontsize=20)
    plt.ylabel('$C_p$' ,fontsize=20)
    #plt.title('%s-AoA-%s-p'%(flist[ii],AoA[jj]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4) 
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.savefig('./plot_ts/%s_%s_aoa_%s-p.pdf'%(i,name[i][0],val_inp[2,0]), format='pdf',bbox_inches='tight', dpi=200)
    plt.show()    
    plt.close()

#plot
def line_plotu_sub(i):
    
    mei=2
    
    plt.figure(figsize=(8, 4), dpi=100)
    
    plt.subplot(1,4,1)
    plt.plot(u1a,ya,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(u2a,ya,'r',linewidth=3,label='NN')
    plt.legend()
    plt.ylabel('Y',fontsize=20)
    plt.xlim(-0.1,1.2)
    
    plt.subplot(1,4,2)
    plt.plot(u1b,yb,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(u2b,yb,'r',linewidth=3)
    plt.xlabel('u-velocity',fontsize=20)
    plt.yticks([])
    plt.xlim(-0.1,1.2)
    
    
    plt.subplot(1,4,3)
    plt.plot(u1d,yd,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(u2d,yd,'r',linewidth=3)
    plt.yticks([])    
    plt.xlim(-0.1,1.2)
    
    plt.subplots_adjust(top = 1, bottom = 0.13, right = 1, left = 0.12, hspace = 0.0, wspace = 0.1)
    plt.savefig('./plot_ts/%s_%s_aoa_%s-u.pdf'%(i,name[i][0],val_inp[2,0]), format='pdf', dpi=200)
    plt.show()   
    plt.close()
    
#plot
def line_plotv_sub(i):
    
    mei=2
    
    plt.figure(figsize=(8, 4), dpi=100)
    
    plt.subplot(1,4,1)
    plt.plot(v1a,ya,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(v2a,ya,'r',linewidth=3,label='NN')
    plt.legend()
    plt.ylabel('Y',fontsize=20)
    plt.xlim(-0.1,1.0)
    
    plt.subplot(1,4,2)
    plt.plot(v1b,yb,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(v2b,yb,'r',linewidth=3)
    plt.xlabel('v-velocity',fontsize=20)
    plt.yticks([])
    plt.xlim(-0.1,0.5)
    
    
    plt.subplot(1,4,3)
    plt.plot(v1d,yd,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(v2d,yd,'r',linewidth=3)
    plt.yticks([])    
    plt.xlim(-0.1,0.5)
       
        
    plt.subplots_adjust(top = 1, bottom = 0.13, right = 1, left = 0.12, hspace = 0, wspace = 0.1)
    plt.savefig('./plot_ts/%s_%s_aoa_%s-v.pdf'%(i,name[i][0],val_inp[2,0]), format='pdf', dpi=200)
    plt.show() 
    plt.close()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

for i in range(175,176):
    
    #normalize
    inp_reno[i]=inp_reno[i]/2000.
    inp_aoa[i]=inp_aoa[i]/14.0
    
    val_inp=np.concatenate((inp_x[i][:,None],inp_y[i][:,None],inp_reno[i][:,None],inp_aoa[i][:,None],inp_para[i][:,:]),axis=1)
    val_out=np.concatenate((out_p[i][:,None],out_u[i][:,None],out_v[i][:,None]),axis=1)


    #load_model
    model_test=load_model('./selected_model/case_8_naca_lam_2/model_sf_140_0.00000278_0.00000300.hdf5') 
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

    a0=find_nearest(co[i][:,0],0)
    
    xu=co[i][:a0+1,0]
    yu=co[i][:a0+1,1]
    if(yu[0] <=0.001):
        yu[0]=0.001
        
    xl=co[i][a0:,0]
    yl=co[i][a0:,1]
    if(yl[-1:] >=-0.001):
        yl[-1:]=-0.001  
    
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
    dl=int(len(co[i][:,0])/2)
    
    a0=find_nearest(co[i][:dl+2,0],0)
    a5=find_nearest(co[i][:dl+2,0],0.5)
    
    xa=np.linspace(co[i][a0,0],co[i][a0,0],50)
    ya=np.linspace(co[i][a0,1],0.5,50)

    xb=np.linspace(co[i][a5,0],co[i][a5,0],50)
    yb=np.linspace(co[i][a5,1],0.5,50)

    xc=np.linspace(co[i][0,0],co[i][0,0],50)
    yc=np.linspace(co[i][0,1],0.5,50)

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
    line_plotu_sub(i)
    line_plotv_sub(i)

