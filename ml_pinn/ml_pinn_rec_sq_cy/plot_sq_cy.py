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
suff='5555_3s_wake-'
for ii in range(len(flist)):
    #x,y,Re,u,v
    xyu_io=np.loadtxt('./data_file/cy_internal_3334.dat',skiprows=1)
    
    xtmp = xyu_io[:,0]
    ytmp = xyu_io[:,1]
    p    = xyu_io[:,2]
    u    = xyu_io[:,3]
    v    = xyu_io[:,4]  

    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
p=np.asarray(p)
u=np.asarray(u)
v=np.asarray(v)

#coord
co=np.zeros((180,2))
x1=np.linspace(-0.5,-0.5,20)
y1=np.linspace(-0.5,0.5,20)
x2=np.linspace(-0.5,0.5,120)
y2=np.linspace(0.5,0.5,120)
x3=np.linspace(0.5,0.5,20)
y3=np.linspace(-0.5,0.5,20)
x4=np.linspace(-0.5,0.5,20)
y4=np.linspace(-0.5,-0.5,20)
co[:,0]=np.concatenate((x1,x2,x3,x4),axis=0)
co[:,1]=np.concatenate((y1,y2,y3,y4),axis=0)

val_inp=np.concatenate((xtmp[:,None],ytmp[:,None]),axis=1)
val_out=np.concatenate((u[:,None],v[:,None],p[:,None]),axis=1)    

#load model
#session-run
tf.reset_default_graph    
graph = tf.get_default_graph() 
#load model
with tf.Session() as sess1:
    
    path1='./tf_model/case_1_4s_around_no_dp/tf_model/'
    new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
    new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))

    tf_dict = {'input0:0': xtmp[:,None], 'input1:0': ytmp[:,None]}

    op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
    
    #uvp
    tout = sess1.run(op_to_load1, tf_dict)

sess1.close()

p_pred=tout[:,2]
u_pred=tout[:,0]
v_pred=tout[:,1]

kout=val_out
i=1
j=1    
   
#open pdf file
fp= PdfPages('plots_turb_ts_1.pdf')

#plot
def con_plot():
    
    fig = plt.figure(figsize=(10, 12),dpi=100)
    
    
    l1=-3
    l2=4
    h1=-3
    h2=3
    
    lp=np.linspace(p.min(),p.max(),20)    
    lu=np.linspace(u.min(),u.max(),20)      
    lv=np.linspace(v.min(),v.max(),20)   
    
    ax1 = fig.add_subplot(3,2,1)
    cp1 = ax1.tricontourf(xtmp,ytmp,p,levels=lp,cmap=cm.jet,extend ='both')
    ax1.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax1.set_title('p-CFD')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1=plt.colorbar(cp1, cax=cax, orientation='vertical');
    cbar1.ax.tick_params(labelsize=10)
    #plt.subplots_adjust( wspace=0.2,hspace=0.3)
    ax1.set_aspect(0.9)
    
    ax2 = fig.add_subplot(3,2,2)
    cp2 = ax2.tricontourf(xtmp,ytmp,p_pred,levels=lp,cmap=cm.jet,extend ='both')
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
    cp3 = ax3.tricontourf(xtmp,ytmp,u,levels=lu,cmap=cm.jet,extend ='both')
    ax3.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax3.set_title('u-CFD')
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
    cp4 = ax4.tricontourf(xtmp,ytmp,u_pred,levels=lu,cmap=cm.jet,extend ='both')
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
    cp5 = ax5.tricontourf(xtmp,ytmp,v,levels=lv,cmap=cm.jet,extend ='both')
    ax5.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax5.set_title('v-CFD')
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
    cp6 = ax6.tricontourf(xtmp,ytmp,v_pred,levels=lv,cmap=cm.jet,extend ='both')
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
    
    #fig.suptitle(" Re=40 at t=0", fontsize=20)
    
    plt.subplots_adjust(top = 0.98, bottom = 0.05, right = 0.9, left = 0.1, hspace = 0.2, wspace = 0.2)       
    plt.savefig('./plot/cy_%s.png'%suff,format='png',dpi=300)
    plt.close()

con_plot()    

def plot_cp():
        
    #LinearNDinterpolator
    pD=np.asarray([xtmp,ytmp]).transpose()

    a0=40
    
    xu=co[20:140,0]
    yu=co[20:140,1]
        
    xl=co[60:80,0]
    yl=co[60:80,1]

    #for -p
    print ('interpolation-1...')      
    f1p=interpolate.LinearNDInterpolator(pD,p)
        
    pu1=np.zeros(len(xu))
    for j in range(len(xu)):
        pu1[j]=f1p(xu[j],yu[j])
    pl1=np.zeros(len(xl))
    for j in range(len(xl)):
        pl1[j]=f1p(xl[j],yl[j])
        
    print ('interpolation-2...')      
    f2p=interpolate.LinearNDInterpolator(pD,p_pred)
      
    pu2=np.zeros(len(xu))
    for j in range(len(xu)):
        pu2[j]=f2p(xu[j],yu[j])
    pl2=np.zeros(len(xl))
    for j in range(len(xl)):
       pl2[j]=f2p(xl[j],yl[j])    
    
    mei=2      
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xu,pu1,'og',linewidth=3,markevery=mei,label='CFD')
    #plt.plot(xl,pl1,'ob',linewidth=3,markevery=mei,label='CFD-lower') 
    plt.plot(xu,pu2,'r',linewidth=3,label='PINN')
    #plt.plot(xl,pl2,'k',linewidth=3,label='NN-lower')  
    plt.xlabel('X',fontsize=20)
    plt.ylabel('P',fontsize=20)
    plt.title('Pressure Dist. over cylinder')
    plt.legend(fontsize=14)
    plt.savefig('./plot/cp_%s.png'%suff,format='png',bbox_inches='tight', dpi=100)
    plt.show()
    
plot_cp()

    
#plot
def line_plotu_sub(i):

    #LinearNDinterpolator
    pD=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()
    
    xa=np.linspace(-0.25,0.25,50)
    ya=np.linspace(0,0.5,50)

    xb=np.linspace(0,0,50)
    yb=np.linspace(0.5,1.0,50)

    xc=np.linspace(1,1,50)
    yc=np.linspace(-0,1,50)

    xd=np.linspace(1.5,1.5,50)
    yd=np.linspace(-0,1,50)
            
    #for u    
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

    print 'interpolation-2...'      
    f3v=interpolate.LinearNDInterpolator(pD,kout[:,1])
   
    v3a=np.zeros((len(ya)))
    v3b=np.zeros((len(ya)))
    v3c=np.zeros((len(ya)))
    v3d=np.zeros((len(ya)))
    for j in range(len(ya)):
        v3a[j]=f3v(xa[j],ya[j])
        v3b[j]=f3v(xb[j],yb[j])
        v3c[j]=f3v(xc[j],yc[j])
        v3d[j]=f3v(xd[j],yd[j])
  
    mei=2
    
    plt.figure(figsize=(6, 4), dpi=100)
    
    plt.subplot(1,3,1)
    plt.plot(u1b,yb,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(u2b,yb,'g',linewidth=3,label='PINN')
    #plt.plot(u3a,ya,'r',linewidth=3,label='NN')
    #plt.legend(fontsize=14)
    plt.ylabel('Y',fontsize=20)
    #plt.xlim(-0.1,1.2)
    plt.ylim(-0.05,1.05)    
    
    plt.subplot(1,3,2)
    plt.plot(u1c,yc,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(u2c,yc,'g',linewidth=3)
    #plt.plot(u3b,yb,'r',linewidth=3)
    plt.xlabel('u-velocity',fontsize=20)
    plt.yticks([])
    #plt.xlim(-0.1,1.2)
    plt.ylim(-0.05,1.05)    
        
    plt.subplot(1,3,3)
    plt.plot(u1d,yd,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(u2d,yd,'g',linewidth=3,label='PINN')
    #plt.plot(u3d,yd,'r',linewidth=3,label='NN')
    plt.legend(loc="upper left", bbox_to_anchor=[-0.02, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
    plt.yticks([])    
    #plt.xlim(-0.5,1.2)
    plt.ylim(-0.05,1.05)    
    
    plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
    plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
    plt.savefig('./plot/u_%s_%s.png'%(i,suff), format='png', bbox_inches='tight',dpi=100)
    plt.show()   
    plt.close()
    
    
    mei=2
    
    plt.figure(figsize=(6, 4), dpi=100)
    
    plt.subplot(1,3,1)
    plt.plot(v1b,yb,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(v2b,yb,'g',linewidth=3)
    #plt.plot(v3a,ya,'r',linewidth=3)
    #plt.legend(fontsize=14)
    plt.ylabel('Y',fontsize=20)
    #plt.xlim(-0.1,1.0)
    plt.ylim(-0.05,1.05)    
    
    plt.subplot(1,3,2)
    plt.plot(v1c,yc,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(v2c,yc,'g',linewidth=3)
    #plt.plot(v3b,yb,'r',linewidth=3)
    plt.xlabel('v-velocity',fontsize=20)
    plt.yticks([])
    #plt.xlim(-0.1,0.5)
    plt.ylim(-0.05,1.05)        
    
    plt.subplot(1,3,3)
    plt.plot(v1d,yd,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(v2d,yd,'g',linewidth=3,label='PINN')
    #plt.plot(v3d,yd,'r',linewidth=3,label='NN')
    plt.yticks([])  
    plt.legend(loc="upper left", bbox_to_anchor=[0.19, 0.5], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
    plt.ylim(-0.05,1.05)    
    #plt.xlim(-0.3,0.5)
       
    plt.figtext(0.4, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
    plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
    plt.savefig('./plot/v_%s_%s.png'%(i,suff), format='png',bbox_inches='tight', dpi=100)
    plt.show() 
    plt.close()    
    
line_plotu_sub(j)



    
fp.close()    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
