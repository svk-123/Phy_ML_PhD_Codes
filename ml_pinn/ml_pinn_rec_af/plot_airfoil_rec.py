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
import tensorflow as tf
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

#1,12,19,20,23
xy=np.loadtxt('./data_file/naca4412_internal_combined_1211.dat',skiprows=1)

val_inp=np.concatenate((xy[:,0:1],xy[:,1:2]),axis=1)
val_out=np.concatenate((xy[:,3:4],xy[:,4:5],xy[:,2:3]),axis=1)    

xtmp=xy[:,0]
ytmp=xy[:,1]
ptmp=xy[:,2]
utmp=xy[:,3]
vtmp=xy[:,4]

#load model
#session-run
tf.reset_default_graph    
graph = tf.get_default_graph() 
#load model
with tf.Session() as sess1:
    
    path1='./tf_model/case_1_naca4412_Re100_aoa20_nodp_nodv_ws_ar_30x2/tf_model/'
    #path1='./tf_model/'
    new_saver1 = tf.train.import_meta_graph( path1 + 'model_50000.meta')
    new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))

    tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
               'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }
    
    op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
    
    #uvp
    tout = sess1.run(op_to_load1, tf_dict)

sess1.close()


co=np.loadtxt('./data_file/naca4412_200_cos.dat',skiprows=1)  
i=1
j=1
  

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#plot
def con_plot(i):
    l1=-1
    l2=2
    h1=-1
    h2=1
    
    fig = plt.figure(figsize=(10, 12),dpi=100)
        
    ax1 = fig.add_subplot(3,2,1)
    cp1 = ax1.tricontourf(xtmp,ytmp,ptmp,20,cmap=cm.jet)
    ax1.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax1.set_title('p-cfd')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1=plt.colorbar(cp1, cax=cax, orientation='vertical');
    cbar1.ax.tick_params(labelsize=10)
    plt.subplots_adjust( wspace=0.2,hspace=0.3)
    ax1.set_aspect(0.9)
    
    ax2 = fig.add_subplot(3,2,2)
    cp2 = ax2.tricontourf(xtmp,ytmp,tout[:,2],20,cmap=cm.jet)
    ax2.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax2.set_title('p-PINN')
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_yticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2=plt.colorbar(cp2, cax=cax, orientation='vertical');
    cbar2.ax.tick_params(labelsize=10)
    ax2.set_aspect(0.9)
      
            
    ax3 = fig.add_subplot(3,2,3)
    cp3 = ax3.tricontourf(xtmp,ytmp,utmp,20,cmap=cm.jet)
    ax3.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax3.set_title('u-cfd')
    ax3.set_xlabel('X',fontsize=16)
    ax3.set_ylabel('Y',fontsize=16)
    ax3.set_xlim([l1,l2])
    ax3.set_ylim([h1,h2])
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3=plt.colorbar(cp3, cax=cax, orientation='vertical');
    cbar3.ax.tick_params(labelsize=10)
    ax3.set_aspect(0.9)
        
    ax4 = fig.add_subplot(3,2,4)
    cp4 = ax4.tricontourf(xtmp,ytmp,tout[:,0],20,cmap=cm.jet)
    ax4.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax4.set_title('u-PINN')
    ax4.set_xlabel('X',fontsize=16)
    ax4.set_yticks([])
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    ax4.set_aspect(0.9)      

         
    ax5 = fig.add_subplot(3,2,5)
    cp5 = ax5.tricontourf(xtmp,ytmp,vtmp,20,cmap=cm.jet)
    ax5.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax5.set_title('v-cfd')
    ax5.set_xlabel('X',fontsize=16)
    ax5.set_ylabel('Y',fontsize=16)
    ax5.set_xlim([l1,l2])
    ax5.set_ylim([h1,h2])
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar5=plt.colorbar(cp5, cax=cax, orientation='vertical');
    cbar5.ax.tick_params(labelsize=10)
    ax5.set_aspect(0.9)
        
    ax6 = fig.add_subplot(3,2,6)
    cp6 = ax6.tricontourf(xtmp,ytmp,tout[:,1],20,cmap=cm.jet)
    ax6.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax6.set_title('v-PINN')
    ax6.set_xlabel('X',fontsize=16)
    ax6.set_yticks([])
    ax6.set_xlim([l1,l2])
    ax6.set_ylim([h1,h2])
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar6=plt.colorbar(cp6, cax=cax, orientation='vertical');
    cbar6.ax.tick_params(labelsize=10)
    ax6.set_aspect(0.9)

   
    #fig.suptitle(" NACA 0012 fontsize=20)
    
    plt.subplots_adjust( wspace=0.2,hspace=0.2)       
    plt.savefig('./plot/airfoil_%s.png'%(i),format='png',dpi=300)
    plt.close()

con_plot(j)  


def plot_cp(i):
        
    #LinearNDinterpolator
    pD=np.asarray([xtmp,ytmp]).transpose()

    a0=find_nearest(co[:,0],0)
    
    xu=co[:a0+1,0]
    yu=co[:a0+1,1]

        
    xl=co[a0:,0]
    yl=co[a0:,1]
             
    #for -p
    print ('interpolation-1...')      
    f1p=interpolate.LinearNDInterpolator(pD,ptmp)
        
    pu1=np.zeros(len(xu))
    for j in range(len(xu)):
        pu1[j]=f1p(xu[j],yu[j])
    pl1=np.zeros(len(xl))
    for j in range(len(xl)):
        pl1[j]=f1p(xl[j],yl[j])
        
    print ('interpolation-2...')      
    f2p=interpolate.LinearNDInterpolator(pD,tout[:,2])
      
    pu2=np.zeros(len(xu))
    for j in range(len(xu)):
        pu2[j]=f2p(xu[j],yu[j])
    pl2=np.zeros(len(xl))
    for j in range(len(xl)):
       pl2[j]=f2p(xl[j],yl[j])    


#    print ('interpolation-2...')      
#    f3p=interpolate.LinearNDInterpolator(pD,kout[:,2])
#      
#    pu3=np.zeros(len(xu))
#    for j in range(len(xu)):
#        pu3[j]=f3p(xu[j],yu[j])
#    pl3=np.zeros(len(xl))
#    for j in range(len(xl)):
#       pl3[j]=f3p(xl[j],yl[j])  

    
    mei=5       
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xu,pu1,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD-upper')
    plt.plot(xl,pl1,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD-lower') 
    plt.plot(xu,pu2,'g',linewidth=3,label='PINN-upper')
    plt.plot(xl,pl2,'g',linewidth=3,label='PINN-lower')  
    #plt.plot(xu,pu3,'r',linewidth=3,label='NN-upper')
    #plt.plot(xl,pl3,'r',linewidth=3,label='NN-lower')  
    plt.xlabel('X',fontsize=20)
    plt.ylabel('P',fontsize=20)
    plt.title('Pressure Dist. over airfoil')
    plt.legend(fontsize=14)
    plt.savefig('./plot/cp_%s.png'%(i),format='png',bbox_inches='tight', dpi=100)
    plt.show()
    
plot_cp(j)  
    
#plot
def line_plotu_sub(i):


    #LinearNDinterpolator
    pD=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()

    a0=find_nearest(co[:,0],0)
    
    #plot -u,v
    dl=int(len(co[:,0])/2)
    
    a0=find_nearest(co[:dl+2,0],0)
    a5=find_nearest(co[:dl+2,0],0.5)
    
    xa=np.linspace(co[a0,0],co[a0,0],50)
    ya=np.linspace(co[a0,1],0.5,50)

    xb=np.linspace(co[a5,0],co[a5,0],50)
    yb=np.linspace(co[a5,1],0.5,50)

    xc=np.linspace(co[0,0],co[0,0],50)
    yc=np.linspace(co[0,1],0.5,50)

    xd=np.linspace(1.2,1.2,50)
    yd=np.linspace(co[0,1],0.99,50)
        
        
    # for u    
    print 'interpolation-1...'      
    f1u=interpolate.LinearNDInterpolator(pD,val_out[:,0])
        
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
    f2u=interpolate.LinearNDInterpolator(pD,tout[:,0])
        
    u2a=np.zeros((len(ya)))
    u2b=np.zeros((len(ya)))
    u2c=np.zeros((len(ya)))
    u2d=np.zeros((len(ya)))
    for j in range(len(ya)):
        u2a[j]=f2u(xa[j],ya[j])
        u2b[j]=f2u(xb[j],yb[j])
        u2c[j]=f2u(xc[j],yc[j])
        u2d[j]=f2u(xd[j],yd[j])

#    print 'interpolation-2...'      
#    f3u=interpolate.LinearNDInterpolator(pD,kout[:,0])
#        
#    u3a=np.zeros((len(ya)))
#    u3b=np.zeros((len(ya)))
#    u3c=np.zeros((len(ya)))
#    u3d=np.zeros((len(ya)))
#    for j in range(len(ya)):
#        u3a[j]=f3u(xa[j],ya[j])
#        u3b[j]=f3u(xb[j],yb[j])
#        u3c[j]=f3u(xc[j],yc[j])
#        u3d[j]=f3u(xd[j],yd[j])


    #for -v
    print 'interpolation-1...'      
    f1v=interpolate.LinearNDInterpolator(pD,val_out[:,1])

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
    f2v=interpolate.LinearNDInterpolator(pD,tout[:,1])
   
    v2a=np.zeros((len(ya)))
    v2b=np.zeros((len(ya)))
    v2c=np.zeros((len(ya)))
    v2d=np.zeros((len(ya)))
    for j in range(len(ya)):
        v2a[j]=f2v(xa[j],ya[j])
        v2b[j]=f2v(xb[j],yb[j])
        v2c[j]=f2v(xc[j],yc[j])
        v2d[j]=f2v(xd[j],yd[j])

#    print 'interpolation-2...'      
#    f3v=interpolate.LinearNDInterpolator(pD,kout[:,1])
#   
#    v3a=np.zeros((len(ya)))
#    v3b=np.zeros((len(ya)))
#    v3c=np.zeros((len(ya)))
#    v3d=np.zeros((len(ya)))
#    for j in range(len(ya)):
#        v3a[j]=f3v(xa[j],ya[j])
#        v3b[j]=f3v(xb[j],yb[j])
#        v3c[j]=f3v(xc[j],yc[j])
#        v3d[j]=f3v(xd[j],yd[j])
  
    mei=2
    
    plt.figure(figsize=(6, 4), dpi=100)
    
    plt.subplot(1,3,1)
    plt.plot(u1a,ya,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(u2a,ya,'g',linewidth=3,label='PINN')
    #plt.plot(u3a,ya,'r',linewidth=3,label='NN')
    #plt.legend(fontsize=14)
    plt.ylabel('Y',fontsize=20)
    plt.xlim(-0.1,1.2)
    
    plt.subplot(1,3,2)
    plt.plot(u1b,yb,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(u2b,yb,'g',linewidth=3)
    #plt.plot(u3b,yb,'r',linewidth=3)
    plt.xlabel('u-velocity',fontsize=20)
    plt.yticks([])
    plt.xlim(-0.1,1.2)
        
    plt.subplot(1,3,3)
    plt.plot(u1d,yd,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(u2d,yd,'g',linewidth=3,label='PINN')
    #plt.plot(u3d,yd,'r',linewidth=3,label='NN')
    plt.legend(loc="upper left", bbox_to_anchor=[-0.02, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
    plt.yticks([])    
    plt.xlim(-0.5,1.2)
    
    plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
    plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
    plt.savefig('./plot/u_%s.png'%(i), format='png', bbox_inches='tight',dpi=100)
    plt.show()   
    plt.close()
    
    
    mei=2
    
    plt.figure(figsize=(6, 4), dpi=100)
    
    plt.subplot(1,3,1)
    plt.plot(v1a,ya,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(v2a,ya,'g',linewidth=3)
    #plt.plot(v3a,ya,'r',linewidth=3)
    #plt.legend(fontsize=14)
    plt.ylabel('Y',fontsize=20)
    plt.xlim(-0.1,1)
    
    plt.subplot(1,3,2)
    plt.plot(v1b,yb,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(v2b,yb,'g',linewidth=3)
    #plt.plot(v3b,yb,'r',linewidth=3)
    plt.xlabel('v-velocity',fontsize=20)
    plt.yticks([])
    plt.xlim(-0.1,0.5)
    
    
    plt.subplot(1,3,3)
    plt.plot(v1d,yd,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(v2d,yd,'g',linewidth=3,label='PINN')
    #plt.plot(v3d,yd,'r',linewidth=3,label='NN')
    plt.yticks([])  
    plt.legend(loc="upper left", bbox_to_anchor=[0.19, 0.5], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')

    plt.xlim(-0.15,0.5)
       
    plt.figtext(0.4, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
    plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
    plt.savefig('./plot/v_%s.png'%(i), format='png',bbox_inches='tight', dpi=100)
    plt.show() 
    plt.close()    
    
line_plotu_sub(j)

    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
