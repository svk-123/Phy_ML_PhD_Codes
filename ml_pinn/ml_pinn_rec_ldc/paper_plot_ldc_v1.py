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



Re=100
suff='%s'%Re    
xy=np.loadtxt('./data_file/Re%s/ldc_internal_combined.dat'%Re,skiprows=1)

val_inp=np.concatenate((xy[:,0:1],xy[:,1:2]),axis=1)
val_out=np.concatenate((xy[:,3:4],xy[:,4:5],xy[:,2:3]),axis=1)    

xtmp=xy[:,0]
ytmp=xy[:,1]
p=xy[:,2]
u=xy[:,3]
v=xy[:,4]

#load model
#session-run
tf.reset_default_graph    
graph = tf.get_default_graph() 
#load model
with tf.Session() as sess1:
    
    path1='./tf_model/case_1_Re%s_nodp_nodv_ws_5x5/tf_model/'%Re
    new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
    new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))

    tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
               'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }

    op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
    
    #uvp
    tout = sess1.run(op_to_load1, tf_dict)

sess1.close()

p_pred=tout[:,2]
u_pred=tout[:,0]
v_pred=tout[:,1]




def con_plot2():
    
    suff='Re100'
    
    l1=0
    l2=1
    h1=0
    h2=1
    
    a1=0.35
    b1=-0.1
    a2=0.35
    b2=-0.1    
    
    AR=1
    
    fig = plt.figure(figsize=(7, 6),dpi=100)

        
    lp=np.linspace(p.min(),p.max(),20)    
    lu=np.linspace(u.min(),u.max(),20)      
    lv=np.linspace(v.min(),v.max(),20)   
    
    lpa=np.linspace(p.min(),p.max(),30)    
    lua=np.linspace(u.min(),u.max(),30)      
    lva=np.linspace(v.min(),v.max(),30)     
    
    ax1 = fig.add_subplot(2,2,1)
    cp1a = ax1.tricontour(xtmp,ytmp,u,levels=lua,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,u,levels=lu,cmap=cm.jet,extend ='both')

    
    #ax1.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    #ax1.set_title('p-CFD')
    ax1.set_xticks([])
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(a1,b1,'(a) u-CFD',fontsize=14)
#    divider = make_axes_locatable(ax1)
#    cax = divider.append_axes('right', size='1%', pad=0.05)
#    cbar1=plt.colorbar(cp1, cax=cax, orientation='vertical')
#    cbar1.ax.tick_params(labelsize=10)
    ax1.set_aspect(AR)       
    
    ax2 = fig.add_subplot(2,2,2)
    cp2a = ax2.tricontour(xtmp,ytmp,u_pred,levels=lua,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,u_pred,levels=lu,cmap=cm.jet,extend ='both')
    #ax2.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    #ax2.set_title('p-PINN')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(a2,b2,'(b) u-PINN',fontsize=14)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    cbar2=plt.colorbar(cp2, cax=cax, orientation='vertical');
    cbar2.ax.tick_params(labelsize=10)
    ax2.set_aspect(AR)
        
    ax3 = fig.add_subplot(2,2,3)
    cp3a = ax3.tricontour(xtmp,ytmp,v,levels=lva,linewidths=0.4,colors='k',zorder=5)    
    cp3 = ax3.tricontourf(xtmp,ytmp,v,levels=lv,cmap=cm.jet,extend ='both')
    #ax3.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    #ax3.set_title('u-CFD')
    ax3.set_xticks([])
    ax3.set_ylabel('Y',fontsize=16)
    ax3.set_xlim([l1,l2])
    ax3.set_ylim([h1,h2])
    ax3.text(a1,b1,'(c) v-CFD',fontsize=14)    
#    divider = make_axes_locatable(ax3)
#    cax = divider.append_axes('right', size='1%', pad=0.05)
#    cbar3=plt.colorbar(cp3, cax=cax, orientation='vertical');
#    cbar3.ax.tick_params(labelsize=10)
    ax3.set_aspect(AR)
        
    ax4 = fig.add_subplot(2,2,4)
    cp4a = ax4.tricontour(xtmp,ytmp,v_pred,levels=lva,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,v_pred,levels=lv,cmap=cm.jet,extend ='both')
    #ax4.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    #ax4.set_title('u-PINN')
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(a2,b2,'(d) v-PINN',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    ax4.set_aspect(AR)      
        
   
    #fig.suptitle(" Re=40 at t=0", fontsize=20)
    
    plt.tight_layout()
    plt.subplots_adjust( hspace = 0.2, wspace = 0.05)       
    plt.savefig('./plot/ldc_%s.tiff'%suff,format='tiff',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()

#con_plot2()    



  

#plot
def line_plot1():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(u1a,ya,'k',marker='o',mew=1.5, mfc='None',ms=12,lw=0,markevery=1,label='CFD')
    plt0, =plt.plot(u2a,ya,'g',linewidth=3,label='PINN')
    plt.legend(fontsize=20,fancybox=False, shadow=False,frameon=False)
    plt.xlabel('u-velocity',fontsize=20)
    plt.ylabel('Y',fontsize=20)
    plt.text(0.35,-0.3,'(a)',fontsize=20) 
    #plt.title('%s-u'%(flist[ii]),fontsiuze=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-u.tiff'%(suff), format='tiff',bbox_inches='tight', dpi=300)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(xb,v1a,'k',marker='o',mew=1.5, mfc='None',ms=12,lw=0,markevery=1,label='CFD')
    plt0, =plt.plot(xb,v2a,'g',linewidth=3,label='PINN')    
    plt.legend(fontsize=20,fancybox=False, shadow=False,frameon=False)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('v-velocity' ,fontsize=20)
    plt.text(0.45,-0.38,'(b)',fontsize=20) 
    #plt.title('%s-v'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-v.tiff'%(suff), format='tiff',bbox_inches='tight', dpi=300)
    plt.show()     
    



#LinearNDinterpolator
pD=np.asarray([xtmp,ytmp]).transpose()
    
print ('interpolation-1...')      
f1u=interpolate.LinearNDInterpolator(pD,u)
xa=np.linspace(0.5,0.5,50)
ya=np.linspace(0.01,0.99,50)
xb=ya
yb=xa
u1a=np.zeros((len(ya)))
u1b=np.zeros((len(ya)))
for i in range(len(ya)):
    u1a[i]=f1u(xa[i],ya[i])
    u1b[i]=f1u(xb[i],yb[i])

print ('interpolation-2...')      
f2u=interpolate.LinearNDInterpolator(pD,tout[:,0])

u2a=np.zeros((len(ya)))
u2b=np.zeros((len(ya)))
for i in range(len(ya)):
    u2a[i]=f2u(xa[i],ya[i])
    u2b[i]=f2u(xb[i],yb[i])

print ('interpolation-3...')      
f1v=interpolate.LinearNDInterpolator(pD,v)

v1a=np.zeros((len(ya)))
v1b=np.zeros((len(ya)))
for i in range(len(ya)):
    v1a[i]=f1v(xb[i],yb[i])
    v1b[i]=f1v(xa[i],ya[i])

print ('interpolation-4...')      
f2v=interpolate.LinearNDInterpolator(pD,tout[:,1])

v2a=np.zeros((len(ya)))
v2b=np.zeros((len(ya)))
for i in range(len(ya)):
    v2a[i]=f2v(xb[i],yb[i])
    v2b[i]=f2v(xa[i],ya[i])


line_plot1()
line_plot2()











    
    
    
    
    
    
    
    
    
    
    
    
    
