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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

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

flist=['re40']
for ii in range(len(flist)):
    #x,y,Re,u,v
    with open('./pred/cy/pred_cy_40_around.pkl', 'rb') as infile:
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


co=np.zeros((100,2))
theta=np.linspace(0,360,100)
for i in range(len(theta)):
    co[i,0]= 0.5*np.cos(np.radians(theta[i]))
    co[i,1]= 0.5*np.sin(np.radians(theta[i]))  
    
    
    
   
#open pdf file
fp= PdfPages('plots_turb_ts_1.pdf')

#plot
def con_plot():
    
    fig = plt.figure(figsize=(10, 12),dpi=100)
        
    ax1 = fig.add_subplot(3,2,1)
    cp1 = ax1.tricontourf(xtmp,ytmp,p[:,0],20,cmap=cm.jet)
    ax1.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax1.set_title('p-cfd')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([-3,5])
    ax1.set_ylim([-3,3])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1=plt.colorbar(cp1, cax=cax, orientation='vertical');
    cbar1.ax.tick_params(labelsize=10)
    plt.subplots_adjust( wspace=0.2,hspace=0.3)
    ax1.set_aspect(0.9)
    
    ax2 = fig.add_subplot(3,2,2)
    cp2 = ax2.tricontourf(xtmp,ytmp,p_pred[:,0],20,cmap=cm.jet)
    ax2.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax2.set_title('p-NN')
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_yticks([])
    ax2.set_xlim([-3,5])
    ax2.set_ylim([-3,3])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2=plt.colorbar(cp2, cax=cax, orientation='vertical');
    cbar2.ax.tick_params(labelsize=10)
    ax2.set_aspect(0.9)
        
    ax3 = fig.add_subplot(3,2,3)
    cp3 = ax3.tricontourf(xtmp,ytmp,u[:,0],20,cmap=cm.jet)
    ax3.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax3.set_title('u-cfd')
    ax3.set_xlabel('X',fontsize=16)
    ax3.set_ylabel('Y',fontsize=16)
    ax3.set_xlim([-3,5])
    ax3.set_ylim([-3,3])
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3=plt.colorbar(cp3, cax=cax, orientation='vertical');
    cbar3.ax.tick_params(labelsize=10)
    ax3.set_aspect(0.9)
        
    ax4 = fig.add_subplot(3,2,4)
    cp4 = ax4.tricontourf(xtmp,ytmp,u_pred[:,0],20,cmap=cm.jet)
    ax4.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax4.set_title('u-NN')
    ax4.set_xlabel('X',fontsize=16)
    ax4.set_yticks([])
    ax4.set_xlim([-3,5])
    ax4.set_ylim([-3,3])
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    ax4.set_aspect(0.9)      
        
    ax5 = fig.add_subplot(3,2,5)
    cp5 = ax5.tricontourf(xtmp,ytmp,v[:,0],20,cmap=cm.jet)
    ax5.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax5.set_title('v-cfd')
    ax5.set_xlabel('X',fontsize=16)
    ax5.set_ylabel('Y',fontsize=16)
    ax5.set_xlim([-3,5])
    ax5.set_ylim([-3,3])
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar5=plt.colorbar(cp5, cax=cax, orientation='vertical');
    cbar5.ax.tick_params(labelsize=10)
    ax5.set_aspect(0.9)
        
    ax6 = fig.add_subplot(3,2,6)
    cp6 = ax6.tricontourf(xtmp,ytmp,v_pred[:,0],20,cmap=cm.jet)
    ax6.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax6.set_title('v-NN')
    ax6.set_xlabel('X',fontsize=16)
    ax6.set_yticks([])
    ax6.set_xlim([-3,5])
    ax6.set_ylim([-3,3])
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar6=plt.colorbar(cp6, cax=cax, orientation='vertical');
    cbar6.ax.tick_params(labelsize=10)
    ax6.set_aspect(0.9)
    
    fig.suptitle(" Re=40 at t=0", fontsize=20)
    
    plt.subplots_adjust( wspace=0.2,hspace=0.25)       
    plt.savefig('cy.png',format='png',dpi=300)
    plt.close()


con_plot()    

def plot_cp():
        
    #LinearNDinterpolator
    pD=np.asarray([xtmp,ytmp]).transpose()

    a0=50
    
    xu=co[:a0+1,0]
    yu=co[:a0+1,1]

        
    xl=co[a0:,0]
    yl=co[a0:,1]
 
             

    #for -p
    print ('interpolation-1...')      
    f1p=interpolate.LinearNDInterpolator(pD,p[:,0])
        
    pu1=np.zeros(len(xu))
    for j in range(len(xu)):
        pu1[j]=f1p(xu[j],yu[j])
    pl1=np.zeros(len(xl))
    for j in range(len(xl)):
        pl1[j]=f1p(xl[j],yl[j])
        
    print ('interpolation-2...')      
    f2p=interpolate.LinearNDInterpolator(pD,p_pred[:,0])
      
    pu2=np.zeros(len(xu))
    for j in range(len(xu)):
        pu2[j]=f2p(xu[j],yu[j])
    pl2=np.zeros(len(xl))
    for j in range(len(xl)):
       pl2[j]=f2p(xl[j],yl[j])    
    
    mei=5       
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xu,pu1,'og',linewidth=3,markevery=mei,label='CFD')
    #plt.plot(xl,pl1,'ob',linewidth=3,markevery=mei,label='CFD-lower') 
    plt.plot(xu,pu2,'r',linewidth=3,label='NN')
    #plt.plot(xl,pl2,'k',linewidth=3,label='NN-lower')  
    plt.xlabel('X',fontsize=20)
    plt.ylabel('P',fontsize=20)
    plt.title('Pressure Dist. over cylinder')
    plt.legend(fontsize=14)
    plt.savefig('./plot/cp_re40.png',format='png',bbox_inches='tight', dpi=100)
    plt.show()
    
plot_cp()


    
fp.close()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
