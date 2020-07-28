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

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

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


 
   
#plot
def con_plot2():
    
    Re=20000
    suff='re%s_nodp_nodv_x8'%Re    
    xy=np.loadtxt('./data_file/Re%s/bl_internal_combined.dat'%Re,skiprows=1)
    
    val_inp=np.concatenate((xy[:,0:1],xy[:,1:2]),axis=1)
    val_out=np.concatenate((xy[:,3:4],xy[:,4:5],xy[:,2:3]),axis=1)    
    
    xtmp=xy[:,0]
    ytmp=xy[:,1]
    p=xy[:,2]
    u=xy[:,3]
    v=xy[:,4]
    
    ##load model
    ##session-run
    tf.reset_default_graph    
    graph = tf.get_default_graph() 
    #load model
    with tf.Session() as sess1:
        
        path1='./tf_model/case_1_re%s_nodp_nodv_with_samling_x8/tf_model/'%Re
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
    
    kout=val_out
    i=1
    j=1   
        
        
    
    
    nu_=1.0/float(Re)
    x1=5.0
    Rex1=x1/nu_
    d5=4.91*x1/np.sqrt(Rex1)
    
    fig = plt.figure(figsize=(5.2*2, 2.2*3),dpi=100)
    
    l1=0
    l2=5
    h1=0
    h2=d5
    
    if (Re==1000):
        a1=-0.05
        a2=-0.20
    if (Re==5000):
        a1=-0.02
        a2=-0.08
    if (Re==10000):
        a1=-0.02
        a2=-0.05
    if (Re==20000):
        a1=-0.01
        a2=-0.04        
        
    lp=np.linspace(p.min(),p.max(),20)    
    lu=np.linspace(u.min(),u.max(),20)      
    lv=np.linspace(v.min(),v.max(),20)   
    
    lpa=np.linspace(p.min(),p.max(),30)    
    lua=np.linspace(u.min(),u.max(),30)      
    lva=np.linspace(v.min(),v.max(),30)     
    
    ax1 = fig.add_subplot(3,2,1)
    cp1a = ax1.tricontour(xtmp,ytmp,p,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,p,levels=lp,cmap=cm.jet,extend ='both')

    
    #ax1.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    #ax1.set_title('p-CFD')
    ax1.set_xticks([])
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(2.0,a1,'(a) p-CFD',fontsize=14)
#    divider = make_axes_locatable(ax1)
#    cax = divider.append_axes('right', size='1%', pad=0.05)
#    cbar1=plt.colorbar(cp1, cax=cax, orientation='vertical')
#    cbar1.ax.tick_params(labelsize=10)
    #ax1.set_aspect(AR,adjustable='box-forced')       
    
    ax2 = fig.add_subplot(3,2,2)
    cp2a = ax2.tricontour(xtmp,ytmp,p_pred,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,p_pred,levels=lp,cmap=cm.jet,extend ='both')
    #ax2.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    #ax2.set_title('p-PINN')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(2.0,a1,'(b) p-PINN',fontsize=14)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar2=plt.colorbar(cp2, cax=cax, orientation='vertical');
    cbar2.ax.tick_params(labelsize=10)
    #ax2.set_aspect(AR,adjustable='box-forced')
        
    ax3 = fig.add_subplot(3,2,3)
    cp3a = ax3.tricontour(xtmp,ytmp,u,levels=lua,linewidths=0.4,colors='k',zorder=5)    
    cp3 = ax3.tricontourf(xtmp,ytmp,u,levels=lu,cmap=cm.jet,extend ='both')
    #ax3.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    #ax3.set_title('u-CFD')
    ax3.set_xticks([])
    ax3.set_ylabel('Y',fontsize=16)
    ax3.set_xlim([l1,l2])
    ax3.set_ylim([h1,h2])
    ax3.text(2.0,a1,'(c) u-CFD',fontsize=14)    
#    divider = make_axes_locatable(ax3)
#    cax = divider.append_axes('right', size='1%', pad=0.05)
#    cbar3=plt.colorbar(cp3, cax=cax, orientation='vertical');
#    cbar3.ax.tick_params(labelsize=10)
    #ax3.set_aspect(AR)
        
    ax4 = fig.add_subplot(3,2,4)
    cp4a = ax4.tricontour(xtmp,ytmp,u_pred,levels=lua,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,u_pred,levels=lu,cmap=cm.jet,extend ='both')
    #ax4.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    #ax4.set_title('u-PINN')
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(2.0,a1,'(d) u-PINN',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    #ax4.set_aspect(AR)      
        
    ax5 = fig.add_subplot(3,2,5)
    cp5a = ax5.tricontour(xtmp,ytmp,v,levels=lva,linewidths=0.4,colors='k',zorder=5)      
    cp5 = ax5.tricontourf(xtmp,ytmp,v,levels=lv,cmap=cm.jet,extend ='both')
    #ax5.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    #ax5.set_title('v-CFD')
    ax5.set_xlabel('X',fontsize=16)
    ax5.set_ylabel('Y',fontsize=16)
    ax5.set_xlim([l1,l2])
    ax5.set_ylim([h1,h2])
    ax5.text(2.0,a2,'(e) v-CFD',fontsize=14)     
#    divider = make_axes_locatable(ax5)
#    cax = divider.append_axes('right', size='1%', pad=0.05)
#    cbar5=plt.colorbar(cp5, cax=cax, orientation='vertical');
#    cbar5.ax.tick_params(labelsize=10)
    #ax5.set_aspect(AR)
        
    ax6 = fig.add_subplot(3,2,6)
    cp6a = ax6.tricontour(xtmp,ytmp,v_pred,levels=lva,linewidths=0.4,colors='k',zorder=5)      
    cp6 = ax6.tricontourf(xtmp,ytmp,v_pred,levels=lv,cmap=cm.jet,extend ='both')
    #ax6.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    #ax6.set_title('v-PINN')
    ax6.set_xlabel('X',fontsize=16)
    ax6.set_yticks([])
    ax6.set_xlim([l1,l2])
    ax6.set_ylim([h1,h2])
    ax6.text(2.0,a2,'(f) v-PINN',fontsize=14)      
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar6=plt.colorbar(cp6, cax=cax, orientation='vertical');
    cbar6.ax.tick_params(labelsize=10)
    #ax6.set_aspect(AR)
    
    #fig.suptitle(" Re=40 at t=0", fontsize=20)
    
    plt.tight_layout()
    plt.subplots_adjust( hspace = 0.3, wspace = 0.05)       
    plt.savefig('./plot/bl_%s.tiff'%suff,format='tiff',bbox_inches='tight',dpi=300)
    plt.close()

#con_plot2()    


#plot
def con_plot4():
    
    Re=100
    suff='re%s_nodp_nodv'%Re    
    xy=np.loadtxt('./data_file/Re%s/bl_internal_combined.dat'%Re,skiprows=1)
    
    val_inp=np.concatenate((xy[:,0:1],xy[:,1:2]),axis=1)
    val_out=np.concatenate((xy[:,3:4],xy[:,4:5],xy[:,2:3]),axis=1)    
    
    xtmp=xy[:,0]
    ytmp=xy[:,1]
    p=xy[:,2]
    u=xy[:,3]
    v=xy[:,4]
    

    
    
    ##load model
    ##session-run
    tf.reset_default_graph    
    graph = tf.get_default_graph() 
    #load model
    with tf.Session() as sess1:
        
        path1='./tf_model/case_1_re%s_dp_dv/tf_model/'%Re
        new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
        new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))
    
        tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
                   'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }
    
        op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
        
        #uvp
        tout1 = sess1.run(op_to_load1, tf_dict)
    
    sess1.close()
    
    
       
    tf.reset_default_graph    
    graph = tf.get_default_graph() 
    #load model
    with tf.Session() as sess2:
        
        path1='./tf_model/case_1_re%s_nodp_nodv/tf_model/'%Re
        new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
        new_saver1.restore(sess2, tf.train.latest_checkpoint(path1))
    
        tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
                   'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }
    
        op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
        
        #uvp
        tout2 = sess2.run(op_to_load1, tf_dict)
    
    sess2.close() 
    

    tf.reset_default_graph    
    graph = tf.get_default_graph() 
    #load model
    with tf.Session() as sess3:
        
        path1='./tf_model/case_1_re%s_4s_puv/tf_model/'%Re
        new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
        new_saver1.restore(sess3, tf.train.latest_checkpoint(path1))
    
        tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
                   'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }
    
        op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
        
        #uvp
        tout3 = sess3.run(op_to_load1, tf_dict)
    
    sess3.close() 
        
            
    
    nu_=1.0/float(Re)
    x1=5.0
    Rex1=x1/nu_
    d5=4.91*x1/np.sqrt(Rex1)
    
    fig = plt.figure(figsize=(5.2*4, 2.2*3),dpi=100)
    
    l1=0
    l2=5
    h1=0
    h2=d5
    
    a1=-0.15
    a2=-0.55

    
    lp=np.linspace(p.min(),p.max(),20)    
    lpa=np.linspace(p.min(),p.max(),30) 
   

    ax1 = fig.add_subplot(3,4,1)
    cp1a = ax1.tricontour(xtmp,ytmp,p,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,p,levels=lp,cmap=cm.jet,extend ='both')
    ax1.set_xticks([])
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(2.0,a1,'(a) p-CFD',fontsize=14)
      
    
    ax2 = fig.add_subplot(3,4,2)
    cp2a = ax2.tricontour(xtmp,ytmp,tout1[:,2],levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,tout1[:,2],levels=lp,cmap=cm.jet,extend ='both')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(2.0,a1,'(b) p-PINN-BC1',fontsize=14)


        
    ax3 = fig.add_subplot(3,4,3)
    cp3a = ax3.tricontour(xtmp,ytmp,tout2[:,2],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp3 = ax3.tricontourf(xtmp,ytmp,tout2[:,2],levels=lp,cmap=cm.jet,extend ='both')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlim([l1,l2])
    ax3.set_ylim([h1,h2])
    ax3.text(2.0,a1,'(c) p-PINN-BC2',fontsize=14)    

        
    ax4 = fig.add_subplot(3,4,4)
    cp4a = ax4.tricontour(xtmp,ytmp,tout3[:,2],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,tout3[:,2],levels=lp,cmap=cm.jet,extend ='both')
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(2.0,a1,'(d) p-PINN-BC3',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)

    
    lp=np.linspace(u.min(),u.max(),20)    
    lpa=np.linspace(u.min(),u.max(),30) 
    
    ax1 = fig.add_subplot(3,4,5)
    cp1a = ax1.tricontour(xtmp,ytmp,u,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,u,levels=lp,cmap=cm.jet,extend ='both')
    ax1.set_xticks([])
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(2.0,a1,'(e) u-CFD',fontsize=14)
      
    
    ax2 = fig.add_subplot(3,4,6)
    cp2a = ax2.tricontour(xtmp,ytmp,tout1[:,0],levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,tout1[:,0],levels=lp,cmap=cm.jet,extend ='both')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(2.0,a1,'(f) u-PINN-BC1',fontsize=14)


        
    ax3 = fig.add_subplot(3,4,7)
    cp3a = ax3.tricontour(xtmp,ytmp,tout2[:,0],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp3 = ax3.tricontourf(xtmp,ytmp,tout2[:,0],levels=lp,cmap=cm.jet,extend ='both')
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_ylabel('Y',fontsize=16)
    ax3.set_xlim([l1,l2])
    ax3.set_ylim([h1,h2])
    ax3.text(2.0,a1,'(g) u-PINN-BC2',fontsize=14)    

        
    ax4 = fig.add_subplot(3,4,8)
    cp4a = ax4.tricontour(xtmp,ytmp,tout3[:,0],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,tout3[:,0],levels=lp,cmap=cm.jet,extend ='both')
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(2.0,a1,'(h) u-PINN-BC3',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    


    lp=np.linspace(v.min(),v.max(),20)    
    lpa=np.linspace(v.min(),v.max(),30)     
        
    ax1 = fig.add_subplot(3,4,9)
    cp1a = ax1.tricontour(xtmp,ytmp,v,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,v,levels=lp,cmap=cm.jet,extend ='both')
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(2.0,a2,'(i) v-CFD',fontsize=14)
      
    
    ax2 = fig.add_subplot(3,4,10)
    cp2a = ax2.tricontour(xtmp,ytmp,tout1[:,1],levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,tout1[:,1],levels=lp,cmap=cm.jet,extend ='both')
    ax2.set_yticks([])
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(2.0,a2,'(j) v-PINN-BC1',fontsize=14)


        
    ax3 = fig.add_subplot(3,4,11)
    cp3a = ax3.tricontour(xtmp,ytmp,tout2[:,1],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp3 = ax3.tricontourf(xtmp,ytmp,tout2[:,1],levels=lp,cmap=cm.jet,extend ='both')
    ax3.set_xlabel('X',fontsize=16)
    ax3.set_yticks([])
    ax3.set_xlim([l1,l2])
    ax3.set_ylim([h1,h2])
    ax3.text(2.0,a2,'(k) v-PINN-BC2',fontsize=14)    

        
    ax4 = fig.add_subplot(3,4,12)
    cp4a = ax4.tricontour(xtmp,ytmp,tout3[:,1],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,tout3[:,1],levels=lp,cmap=cm.jet,extend ='both')
    ax4.set_yticks([])
    ax4.set_xlabel('X',fontsize=16)
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(2.0,a2,'(l) v-PINN-BC3',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    
                         
    
    plt.tight_layout()
    plt.subplots_adjust( hspace = 0.3, wspace = 0.05)       
    plt.savefig('./plot/bl_wos_bc_comp.tiff',format='tiff',bbox_inches='tight',dpi=100)
    plt.close()


#con_plot4()
    
    
    

#plot
def con_plot3():
    
    Re=100
    suff='re%s_nodp_nodv'%Re    
    xy=np.loadtxt('./data_file/Re%s/bl_internal_combined.dat'%Re,skiprows=1)
    
    val_inp=np.concatenate((xy[:,0:1],xy[:,1:2]),axis=1)
    val_out=np.concatenate((xy[:,3:4],xy[:,4:5],xy[:,2:3]),axis=1)    
    
    xtmp=xy[:,0]
    ytmp=xy[:,1]
    p=xy[:,2]
    u=xy[:,3]
    v=xy[:,4]
    
   
    
    ##load model
    ##session-run
    tf.reset_default_graph    
    graph = tf.get_default_graph() 
    #load model
    with tf.Session() as sess1:
        
        path1='./tf_model/case_1_re%s_nodp_nodv_with_samling_x8_away/tf_model/'%Re
        new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
        new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))
    
        tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
                   'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }
    
        op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
        
        #uvp
        tout1 = sess1.run(op_to_load1, tf_dict)
    
    sess1.close()
               
    tf.reset_default_graph    
    graph = tf.get_default_graph() 
    #load model
    with tf.Session() as sess2:
        
        path1='./tf_model/case_1_re%s_nodp_nodv_with_samling_x8/tf_model/'%Re
        new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
        new_saver1.restore(sess2, tf.train.latest_checkpoint(path1))
    
        tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
                   'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }
    
        op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
        
        #uvp
        tout2 = sess2.run(op_to_load1, tf_dict)
    
    sess2.close() 
    
            
    
    nu_=1.0/float(Re)
    x1=5.0
    Rex1=x1/nu_
    d5=4.91*x1/np.sqrt(Rex1)
    
    fig = plt.figure(figsize=(5.2*3, 2.2*3),dpi=100)
    
    l1=0
    l2=5
    h1=0
    h2=d5
    
    a1=-0.15
    a2=-0.55

    
    lp=np.linspace(p.min(),p.max(),20)    
    lpa=np.linspace(p.min(),p.max(),30) 
   

    ax1 = fig.add_subplot(3,3,1)
    cp1a = ax1.tricontour(xtmp,ytmp,p,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,p,levels=lp,cmap=cm.jet,extend ='both')
    ax1.set_xticks([])
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(2.0,a1,'(a) p-CFD',fontsize=14)
      
    
    ax2 = fig.add_subplot(3,3,2)
    cp2a = ax2.tricontour(xtmp,ytmp,tout1[:,2],levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,tout1[:,2],levels=lp,cmap=cm.jet,extend ='both')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(2.0,a1,'(b) p-PINN-S-1',fontsize=14)

        
    ax4 = fig.add_subplot(3,3,3)
    cp4a = ax4.tricontour(xtmp,ytmp,tout2[:,2],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,tout2[:,2],levels=lp,cmap=cm.jet,extend ='both')
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(2.0,a1,'(c) p-PINN-S-2',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)

    
    lp=np.linspace(u.min(),u.max(),20)    
    lpa=np.linspace(u.min(),u.max(),30) 
    
    ax1 = fig.add_subplot(3,3,4)
    cp1a = ax1.tricontour(xtmp,ytmp,u,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,u,levels=lp,cmap=cm.jet,extend ='both')
    ax1.set_xticks([])
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(2.0,a1,'(d) u-CFD',fontsize=14)
      
    
    ax2 = fig.add_subplot(3,3,5)
    cp2a = ax2.tricontour(xtmp,ytmp,tout1[:,0],levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,tout1[:,0],levels=lp,cmap=cm.jet,extend ='both')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(2.0,a1,'(e) u-PINN-S-1',fontsize=14)

        
    ax4 = fig.add_subplot(3,3,6)
    cp4a = ax4.tricontour(xtmp,ytmp,tout2[:,0],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,tout2[:,0],levels=lp,cmap=cm.jet,extend ='both')
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(2.0,a1,'(f) u-PINN-S-2',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    


    lp=np.linspace(v.min(),v.max(),20)    
    lpa=np.linspace(v.min(),v.max(),30)     
        
    ax1 = fig.add_subplot(3,3,7)
    cp1a = ax1.tricontour(xtmp,ytmp,v,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,v,levels=lp,cmap=cm.jet,extend ='both')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(2.0,a2,'(g) u-CFD',fontsize=14)
      
    
    ax2 = fig.add_subplot(3,3,8)
    cp2a = ax2.tricontour(xtmp,ytmp,tout1[:,1],levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,tout1[:,1],levels=lp,cmap=cm.jet,extend ='both')
    ax2.set_yticks([])
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(2.0,a2,'(h) u-PINN-S-1',fontsize=14)

        
    ax4 = fig.add_subplot(3,3,9)
    cp4a = ax4.tricontour(xtmp,ytmp,tout2[:,1],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,tout2[:,1],levels=lp,cmap=cm.jet,extend ='both')
    ax4.set_yticks([])
    ax4.set_xlabel('X',fontsize=16)
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(2.0,a2,'(i) u-PINN-S-2',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    
                             
    plt.tight_layout()
    plt.subplots_adjust( hspace = 0.3, wspace = 0.05)       
    plt.savefig('./plot/bl_ws_s_comp.tiff',format='tiff',bbox_inches='tight',dpi=100)
    plt.close()


#con_plot3()
    
    
    
    
      

#plot
def con_plot3nn():
    
    Re=100
    suff='re%s_nodp_nodv_nopinn'%Re    
    xy=np.loadtxt('./data_file/Re%s/bl_internal_combined.dat'%Re,skiprows=1)
    
    val_inp=np.concatenate((xy[:,0:1],xy[:,1:2]),axis=1)
    val_out=np.concatenate((xy[:,3:4],xy[:,4:5],xy[:,2:3]),axis=1)    
    
    xtmp=xy[:,0]
    ytmp=xy[:,1]
    p=xy[:,2]
    u=xy[:,3]
    v=xy[:,4]
    
   
    
    ##load model
    ##session-run
    tf.reset_default_graph    
    graph = tf.get_default_graph() 
    #load model
    with tf.Session() as sess1:
        
        path1='./tf_model/case_1_re%s_nodp_nodv/tf_model/'%Re
        new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
        new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))
    
        tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
                   'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }
    
        op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
        
        #uvp
        tout1 = sess1.run(op_to_load1, tf_dict)
    
    sess1.close()
               
    tf.reset_default_graph    
    graph = tf.get_default_graph() 
    #load model
    with tf.Session() as sess2:
        
        path1='./tf_model/case_1_re%s_nodp_nodv_nopinn/tf_model/'%Re
        new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
        new_saver1.restore(sess2, tf.train.latest_checkpoint(path1))
    
        tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
                   'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }
    
        op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
        
        #uvp
        tout2 = sess2.run(op_to_load1, tf_dict)
    
    sess2.close() 
    
            
    
    nu_=1.0/float(Re)
    x1=5.0
    Rex1=x1/nu_
    d5=4.91*x1/np.sqrt(Rex1)
    
    fig = plt.figure(figsize=(5.2*3, 2.2*3),dpi=100)
    
    l1=0
    l2=5
    h1=0
    h2=d5
    
    a1=-0.15
    a2=-0.55

    
    lp=np.linspace(p.min(),p.max(),20)    
    lpa=np.linspace(p.min(),p.max(),30) 
   

    ax1 = fig.add_subplot(3,3,1)
    cp1a = ax1.tricontour(xtmp,ytmp,p,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,p,levels=lp,cmap=cm.jet,extend ='both')
    ax1.set_xticks([])
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(2.0,a1,'(a) p-CFD',fontsize=14)
      
    
    ax2 = fig.add_subplot(3,3,2)
    cp2a = ax2.tricontour(xtmp,ytmp,tout1[:,2],levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,tout1[:,2],levels=lp,cmap=cm.jet,extend ='both')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(2.0,a1,'(b) p-PINN',fontsize=14)

        
    ax4 = fig.add_subplot(3,3,3)
    cp4a = ax4.tricontour(xtmp,ytmp,tout2[:,2],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,tout2[:,2],levels=lp,cmap=cm.jet,extend ='both')
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(2.0,a1,'(c) p-NN',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)

    
    lp=np.linspace(u.min(),u.max(),20)    
    lpa=np.linspace(u.min(),u.max(),30) 
    
    ax1 = fig.add_subplot(3,3,4)
    cp1a = ax1.tricontour(xtmp,ytmp,u,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,u,levels=lp,cmap=cm.jet,extend ='both')
    ax1.set_xticks([])
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(2.0,a1,'(d) u-CFD',fontsize=14)
      
    
    ax2 = fig.add_subplot(3,3,5)
    cp2a = ax2.tricontour(xtmp,ytmp,tout1[:,0],levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,tout1[:,0],levels=lp,cmap=cm.jet,extend ='both')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(2.0,a1,'(e) u-PINN',fontsize=14)

        
    ax4 = fig.add_subplot(3,3,6)
    cp4a = ax4.tricontour(xtmp,ytmp,tout2[:,0],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,tout2[:,0],levels=lp,cmap=cm.jet,extend ='both')
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(2.0,a1,'(f) u-NN',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    


    lp=np.linspace(v.min(),v.max(),20)    
    lpa=np.linspace(v.min(),v.max(),30)     
        
    ax1 = fig.add_subplot(3,3,7)
    cp1a = ax1.tricontour(xtmp,ytmp,v,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp1 = ax1.tricontourf(xtmp,ytmp,v,levels=lp,cmap=cm.jet,extend ='both')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    ax1.text(2.0,a2,'(g) v-CFD',fontsize=14)
      
    
    ax2 = fig.add_subplot(3,3,8)
    cp2a = ax2.tricontour(xtmp,ytmp,tout1[:,1],levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,tout1[:,1],levels=lp,cmap=cm.jet,extend ='both')
    ax2.set_yticks([])
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.text(2.0,a2,'(h) v-PINN',fontsize=14)

        
    ax4 = fig.add_subplot(3,3,9)
    cp4a = ax4.tricontour(xtmp,ytmp,tout2[:,1],levels=lpa,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,tout2[:,1],levels=lp,cmap=cm.jet,extend ='both')
    ax4.set_yticks([])
    ax4.set_xlabel('X',fontsize=16)
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
    ax4.text(2.0,a2,'(i) v-NN',fontsize=14)     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
    cbar4.ax.tick_params(labelsize=10)
    
                             
    plt.tight_layout()
    plt.subplots_adjust( hspace = 0.3, wspace = 0.05)       
    plt.savefig('./plot/bl_pinn_nn_comp.tiff',format='tiff',bbox_inches='tight',dpi=100)
    plt.close()


con_plot3nn()
    
    
    
    
    
    
    
    
    
    
    
