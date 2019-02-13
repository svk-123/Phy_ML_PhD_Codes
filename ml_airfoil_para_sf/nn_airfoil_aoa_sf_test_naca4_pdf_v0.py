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
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

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
with open('./data_file/foil_naca4_lam_trts_1.pkl', 'rb') as infile:
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
#fxy=result[9]
name=result[9]

inp_x=np.asarray(inp_x)
inp_y=np.asarray(inp_y)
inp_reno=np.asarray(inp_reno)
inp_aoa=np.asarray(inp_aoa)
inp_para=np.asarray(inp_para)

out_p=np.asarray(out_p)
out_u=np.asarray(out_u)
out_v=np.asarray(out_v)


#open pdf file
fp= PdfPages('plots_lam_naca4_ts_1.pdf')

#plot
def con_plot(i):
    
    fig = plt.figure(figsize=(10, 12),dpi=100)
        
    ax1 = fig.add_subplot(3,2,1)
    cp1 = ax1.tricontourf(val_inp[:,0],val_inp[:,1],val_out[:,0],20,cmap=cm.jet)
    ax1.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
    ax1.set_title('p-cfd')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([-0.5, 2])
    ax1.set_ylim([-0.5, 0.5])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1=plt.colorbar(cp1, cax=cax, orientation='vertical');
    cbar1.ax.tick_params(labelsize=10)
    plt.subplots_adjust( wspace=0.2,hspace=0.3)
    ax1.set_aspect(1.1)
    
    ax2 = fig.add_subplot(3,2,2)
    cp2 = ax2.tricontourf(val_inp[:,0],val_inp[:,1],out[:,0],20,cmap=cm.jet)
    ax2.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
    ax2.set_title('p-NN')
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_yticks([])
    ax2.set_xlim([-0.5, 2])
    ax2.set_ylim([-0.5, 0.5])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2=plt.colorbar(cp2, cax=cax, orientation='vertical');
    cbar2.ax.tick_params(labelsize=10)
    ax2.set_aspect(1.1)
    
    
    ax3 = fig.add_subplot(3,2,3)
    cp3 = ax3.tricontourf(val_inp[:,0],val_inp[:,1],val_out[:,1],20,cmap=cm.jet)
    ax3.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
    ax3.set_title('u-cfd')
    ax3.set_xlabel('X',fontsize=16)
    ax3.set_ylabel('Y',fontsize=16)
    ax3.set_xlim([-0.5, 2])
    ax3.set_ylim([-0.5, 0.5])
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3=plt.colorbar(cp3, cax=cax, orientation='vertical');
    cbar3.ax.tick_params(labelsize=10)
    ax3.set_aspect(1.1)
    
    
    ax4 = fig.add_subplot(3,2,4)
    cp4 = ax4.tricontourf(val_inp[:,0],val_inp[:,1],out[:,1],20,cmap=cm.jet)
    ax4.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
    ax4.set_title('u-NN')
    ax4.set_xlabel('X',fontsize=16)
    ax4.set_yticks([])
    ax4.set_xlim([-0.5, 2])
    ax4.set_ylim([-0.5, 0.5])
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    ax4.set_aspect(1.1)      
    
    
    ax5 = fig.add_subplot(3,2,5)
    cp5 = ax5.tricontourf(val_inp[:,0],val_inp[:,1],val_out[:,2],20,cmap=cm.jet)
    ax5.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
    ax5.set_title('v-cfd')
    ax5.set_xlabel('X',fontsize=16)
    ax5.set_ylabel('Y',fontsize=16)
    ax5.set_xlim([-0.5, 2])
    ax5.set_ylim([-0.5, 0.5])
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar5=plt.colorbar(cp5, cax=cax, orientation='vertical');
    cbar5.ax.tick_params(labelsize=10)
    ax5.set_aspect(1.1)
    
    
    ax6 = fig.add_subplot(3,2,6)
    cp6 = ax6.tricontourf(val_inp[:,0],val_inp[:,1],out[:,2],20,cmap=cm.jet)
    ax6.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
    ax6.set_title('v-NN')
    ax6.set_xlabel('X',fontsize=16)
    ax6.set_yticks([])
    ax6.set_xlim([-0.5, 2])
    ax6.set_ylim([-0.5, 0.5])
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar6=plt.colorbar(cp6, cax=cax, orientation='vertical');
    cbar6.ax.tick_params(labelsize=10)
    ax6.set_aspect(1.1)
    
    fig.suptitle("Testing:Airfoil-%s Re=%se4 AoA=%s"%(name[i][0],int(inp_reno[i][0]*10),int(inp_aoa[i][0]*14)), fontsize=24)
    
    plt.subplots_adjust( wspace=0.2,hspace=0.25)       
    fp.savefig(fig)
    plt.close()



def line_plot(i):
    
    
    
    fig = plt.figure(figsize=(10, 16))
    gs = gridspec.GridSpec(3,4)
    
    mei=int(len(co[i])/30)
    if (mei < 1):
        mei=1
           
    ax1 = plt.subplot(gs[0,1:3])
    ax1.plot(xu,pu1,'og',linewidth=3,markevery=mei,label='CFD-upper')
    ax1.plot(xl,pl1,'ob',linewidth=3,markevery=mei,label='CFD-lower') 
    ax1.plot(xu,pu2,'r',linewidth=3,label='NN-upper')
    ax1.plot(xl,pl2,'k',linewidth=3,label='NN-lower')     
    ax1.set_title('Pressure',fontsize=20)

    ax2 = plt.subplot(gs[1,0:1])    
    ax2.plot(u1a,ya,'-og',linewidth=3,label='CFD')
    ax2.plot(u2a,ya,'r',linewidth=3,label='NN')
    
    ax3 = plt.subplot(gs[1,1:2]) 
    ax3.plot(u1b,yb,'-og',linewidth=3)
    ax3.plot(u2b,yb,'r',linewidth=3)
    ax3.set_title('u - vel',fontsize=20)
    
    ax3 = plt.subplot(gs[1,2:3]) 
    ax3.plot(u1c,yc,'-og',linewidth=3)
    ax3.plot(u2c,yc,'r',linewidth=3)

    ax4 = plt.subplot(gs[1,3:4]) 
    ax4.plot(u1d,yd,'-og',linewidth=3)
    ax4.plot(u2d,yd,'r',linewidth=3)
        
    ax5 = plt.subplot(gs[2,0:1])    
    ax5.plot(v1a,ya,'-og',linewidth=3,label='CFD')
    ax5.plot(v2a,ya,'r',linewidth=3,label='NN')
    ax5.set_xlim([-0.1,0.8])
        
    ax6 = plt.subplot(gs[2,1:2]) 
    ax6.plot(v1b,yb,'-og',linewidth=3)
    ax6.plot(v2b,yb,'r',linewidth=3)
    ax6.set_xlim([-0.1,0.25])
    ax6.set_title('v - vel',fontsize=20)
    
    ax7 = plt.subplot(gs[2,2:3]) 
    ax7.plot(v1c,yc,'-og',linewidth=3)
    ax7.plot(v2c,yc,'r',linewidth=3)
    ax7.set_xlim([-0.1,0.2])
    
    ax8 = plt.subplot(gs[2,3:4]) 
    ax8.plot(v1d,yd,'-og',linewidth=3)
    ax8.plot(v2d,yd,'r',linewidth=3)    
    ax8.set_xlim([-0.1,0.2])
     
    fig.suptitle("Testing:Airfoil-%s Re=%se4 AoA=%s"%(name[i][0],int(inp_reno[i][0]*10),int(inp_aoa[i][0]*14)), fontsize=24)
    
    plt.subplots_adjust( wspace=0.2,hspace=0.5)       
    fp.savefig(fig)
    plt.close()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
np.random.seed(1234534)
mylist=np.random.randint(0,143,5)

for i in mylist:
    print i
    #normalize
    inp_reno[i]=inp_reno[i]/2000.
    inp_aoa[i]=inp_aoa[i]/14.0
    
    val_inp=np.concatenate((inp_x[i][:,None],inp_y[i][:,None],inp_reno[i][:,None],inp_aoa[i][:,None],inp_para[i][:,:]),axis=1)
    val_out=np.concatenate((out_p[i][:,None],out_u[i][:,None],out_v[i][:,None]),axis=1)

    #load_model
    model_test=load_model('./selected_model/case_1_naca_1/model_sf_150_0.00002427_0.00002472.hdf5') 
    out=model_test.predict([val_inp]) 
         
    con_plot(i)

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
    
    
#    xu=fxy[i][0]
#    xl=fxy[i][1]
#    yu=fxy[i][2]
#    yl=fxy[i][3]    
    
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
    a8=find_nearest(co[i][:dl+2,0],0.8)    
    
    
    xa=np.linspace(co[i][a0,0],co[i][a0,0],50)
    ya=np.linspace(co[i][a0,1],0.5,50)

    xb=np.linspace(co[i][a5,0],co[i][a5,0],50)
    yb=np.linspace(co[i][a5,1],0.5,50)

    xc=np.linspace(co[i][a8,0],co[i][a8,0],50)
    yc=np.linspace(co[i][a8,1],0.5,50)

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
    
    line_plot(i)
    
    
fp.close()    



