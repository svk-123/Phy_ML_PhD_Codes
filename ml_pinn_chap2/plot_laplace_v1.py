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

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import pickle

import pandas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from scipy import interpolate
from numpy import linalg as LA
import matplotlib

#matplotlib.rc('xtick', labelsize=18) 
#matplotlib.rc('ytick', labelsize=18) 

##load data
#xtmp=[]
#ytmp=[]
#p=[]
#u=[]
#v=[]
#p_pred=[]
#u_pred=[]
#v_pred=[]
#

#flist=['re40']
#suff='bl'
#for ii in range(len(flist)):
#    #x,y,Re,u,v
#    with open('./data_file/BL_0503.pkl', 'rb') as infile:
#        result = pickle.load(infile)
#    xtmp.extend(result[0])
#    ytmp.extend(result[1])
#    p.extend(result[2])
#    u.extend(result[3])
#    v.extend(result[4])    
#
#    

#xtmp=np.asarray(xtmp)
#ytmp=np.asarray(ytmp)
#p=np.asarray(p)
#u=np.asarray(u)
#v=np.asarray(v)

Re=100
suff='laplace'   
xy=np.loadtxt('./data_file/laplace_internal_combined.dat')

val_inp=np.concatenate((xy[:,0:1],xy[:,1:2]),axis=1)
val_out=xy[:,2:3]

xtmp=xy[:,0]
ytmp=xy[:,1]
u=xy[:,2]


#load model
#session-run
tf.reset_default_graph    
graph = tf.get_default_graph() 
#load model
with tf.Session() as sess1:
    
    path1='./tf_model/'
    new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
    new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))

    tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None]}

    op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
    
    #uvp
    tout = sess1.run(op_to_load1, tf_dict)

sess1.close()

u_pred=tout[:,0]

 
kout=tout   
#plot
def con_plot():
    
    fig = plt.figure(figsize=(8, 4),dpi=100)
    
    l1=0
    l2=1
    h1=0
    h2=1
   
    
    ax1 = fig.add_subplot(1,2,1)
    cp1a = ax1.tricontour(xtmp,ytmp,u,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,u,cmap=cm.jet,extend ='both')

    
    #ax1.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax1.set_title('p-CFD')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    
#    divider = make_axes_locatable(ax1)
#    cax = divider.append_axes('right', size='1%', pad=0.05)
#    cbar1=plt.colorbar(cp1, cax=cax, orientation='vertical')
#    cbar1.ax.tick_params(labelsize=10)
    #ax1.set_aspect(AR,adjustable='box-forced')       
    
    ax2 = fig.add_subplot(1,2,2)
    cp2a = ax2.tricontour(xtmp,ytmp,u_pred,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,u_pred,cmap=cm.jet,extend ='both')
    #ax2.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax2.set_title('p-PINN')
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_yticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    cbar2=plt.colorbar(cp2, cax=cax, orientation='vertical');
    cbar2.ax.tick_params(labelsize=10)
    #ax2.set_aspect(AR,adjustable='box-forced')
        
    
    #plt.tight_layout()
    plt.subplots_adjust( hspace = 0.05, wspace = 0.05)       
    plt.savefig('./plot/laplace_%s.png'%suff,format='png',bbox_inches='tight',dpi=300)
    plt.close()


con_plot()    



j=1    
#plot
def line_plotu_sub(i):

    #LinearNDinterpolator
    pD=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()
    yl=1
    
    ya=np.linspace(0.5,0.5,50)
    xa=np.linspace(0,yl,50)

    yb=np.linspace(0.25,0.25,50)
    xb=np.linspace(0,yl,50)

    yc=np.linspace(0.5,0.5,50)
    xc=np.linspace(0,yl,50)

    yd=np.linspace(0.75,0.75,50)
    xd=np.linspace(0,yl,50)
        
        
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

    print 'interpolation-2...'      
    f3u=interpolate.LinearNDInterpolator(pD,kout[:,0])
        
    u3a=np.zeros((len(ya)))
    u3b=np.zeros((len(ya)))
    u3c=np.zeros((len(ya)))
    u3d=np.zeros((len(ya)))
    for j in range(len(ya)):
        u3a[j]=f3u(xa[j],ya[j])
        u3b[j]=f3u(xb[j],yb[j])
        u3c[j]=f3u(xc[j],yc[j])
        u3d[j]=f3u(xd[j],yd[j])

  
    mei=2
    
    plt.figure(figsize=(6, 4), dpi=100)
    
    plt.plot(xb,u1b,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(xb,u2b,'g',linewidth=3,label='PINN')
    
    plt.plot(xc,u1c,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(xc,u2c,'g',linewidth=3)
    
    plt.plot(xd,u1d,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(xd,u2d,'g',linewidth=3,label='PINN')    
    
    #plt.plot(u3a,ya,'r',linewidth=3,label='NN')
    #plt.legend(fontsize=14)
    plt.ylabel('Y',fontsize=20)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.05,1.05)    
    
        
    #plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
    plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
    plt.savefig('./plot/u_%s_%s.png'%(i,suff), format='png', bbox_inches='tight',dpi=100)
    plt.show()   
    plt.close()
 
line_plotu_sub(j)




    
    
    
    
    
    
    
    
    
    
    
    
    
