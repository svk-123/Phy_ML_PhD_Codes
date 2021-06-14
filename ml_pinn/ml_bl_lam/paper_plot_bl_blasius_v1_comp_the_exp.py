#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon May  1 08:09:04 2017

@author: vinoth

write for blasius velocity profile
write for contours

"""

import time
start_time = time.time()

# Python 3.5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os import listdir
from os.path import isfile, join

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import pickle

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from scipy import interpolate
from numpy import linalg as LA
import matplotlib

#matplotlib.rc('xtick', labelsize=18) 
#matplotlib.rc('ytick', labelsize=18) 

#blasius solution analytical


bla=np.loadtxt('Bla_velocity.dat')
bla=bla[:35,:]








Re=100
suff='re%s_sample_p8'%Re    
xy=np.loadtxt('./data_file/Re%s/bl_internal_combined.dat'%Re,skiprows=1)

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
    
    path1='./tf_model/case_1_re100_nodp_nodv_with_samling_x8/tf_model/'
    new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
    new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))

    #new with nx ny
    tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
                'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }

    '''#old-with out nx ny
    tf_dict = {'input0:0': xtmp[:,None], 'input1:0': ytmp[:,None] }'''
    
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
   
   
#get d1,d2,d3,d4 from CFD
def line_plotu_sub1(i):

    ######-BL thickness--######
    ########################
           
    #LinearNDinterpolator
    pD=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()
        
    # for u    
    print ('interpolation-1...')      
    f1u=interpolate.LinearNDInterpolator(pD,val_out[:,0])
    
    N=300
    
    xa=np.linspace(1,1,N)
    ya=np.linspace(0,0.5,N)  
    
    xb=np.linspace(2,2,N)
    yb=np.linspace(0,1,N)       

    xc=np.linspace(3,3,N)
    yc=np.linspace(0,1.5,N)   
    
    xd=np.linspace(4.9,4.9,N)
    yd=np.linspace(0,1.5,N)       
    
    u1a=np.zeros((len(ya)))
    u2a=np.zeros((len(ya)))
    u3a=np.zeros((len(ya)))
    u4a=np.zeros((len(ya)))
    
    for j in range(len(ya)):
        u1a[j]=f1u(xa[j],ya[j])
        u2a[j]=f1u(xb[j],yb[j])
        u3a[j]=f1u(xc[j],yc[j])
        u4a[j]=f1u(xd[j],yd[j])
               
    c=0
    for k in range(len(ya)):
        if(u1a[k]>=0.99 and c==0):
            d1=ya[k]
            c=1
    c=0
    for k in range(len(ya)):
        if(u2a[k]>=0.99 and c==0):
            d2=yb[k]
            c=1
    c=0
    for k in range(len(ya)):
        if(u3a[k]>=0.99 and c==0):
            d3=yc[k]
            c=1            
            
    c=0
    for k in range(len(ya)):
        if(u4a[k]>=0.99 and c==0):
            d4=yd[k]
            c=1
            
    global d1
    global d2
    global d3
    global d4            
line_plotu_sub1(j)  

#plot
def con_plot():
    
    nu_=1.0/float(Re)
    x1=5
    Rex1=x1/nu_
    d5=4.91*x1/np.sqrt(Rex1)


    
    fig = plt.figure(figsize=(8, 8),dpi=100)
    
    l1=0
    l2=5
    h1=0
    h2=d5
    AR=2.0/h2
     
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
    ax1.set_title('p-CFD')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([l1,l2])
    ax1.set_ylim([h1,h2])
    
#    divider = make_axes_locatable(ax1)
#    cax = divider.append_axes('right', size='1%', pad=0.05)
#    cbar1=plt.colorbar(cp1, cax=cax, orientation='vertical')
#    cbar1.ax.tick_params(labelsize=10)
    ax1.set_aspect(AR,adjustable='box-forced')       
    
    ax2 = fig.add_subplot(3,2,2)
    cp2a = ax2.tricontour(xtmp,ytmp,p_pred,levels=lpa,linewidths=0.4,colors='k',zorder=5)
    cp2 = ax2.tricontourf(xtmp,ytmp,p_pred,levels=lp,cmap=cm.jet,extend ='both')
    #ax2.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax2.set_title('p-PINN')
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_yticks([])
    ax2.set_xlim([l1,l2])
    ax2.set_ylim([h1,h2])
    ax2.set_aspect(AR,adjustable='box-forced')
    
    plt.colorbar(cp2)
#    divider = make_axes_locatable(ax2)
#    cax = divider.append_axes('right', size='5%', pad=0.05)
#    cbar2=plt.colorbar(cp2, cax=cax, orientation='vertical');
#    cbar2.ax.tick_params(labelsize=10)

        
    ax3 = fig.add_subplot(3,2,3)
    cp3a = ax3.tricontour(xtmp,ytmp,u,levels=lua,linewidths=0.4,colors='k',zorder=5)    
    cp3 = ax3.tricontourf(xtmp,ytmp,u,levels=lu,cmap=cm.jet,extend ='both')
    #ax3.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax3.set_title('u-CFD')
    ax3.set_xlabel('X',fontsize=16)
    ax3.set_ylabel('Y',fontsize=16)
    ax3.set_xlim([l1,l2])
    ax3.set_ylim([h1,h2])
#    divider = make_axes_locatable(ax3)
#    cax = divider.append_axes('right', size='1%', pad=0.05)
#    cbar3=plt.colorbar(cp3, cax=cax, orientation='vertical');
#    cbar3.ax.tick_params(labelsize=10)
    ax3.set_aspect(AR)
        
    ax4 = fig.add_subplot(3,2,4)
    cp4a = ax4.tricontour(xtmp,ytmp,u_pred,levels=lua,linewidths=0.4,colors='k',zorder=5)    
    cp4 = ax4.tricontourf(xtmp,ytmp,u_pred,levels=lu,cmap=cm.jet,extend ='both')
    #ax4.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax4.set_title('u-PINN')
    ax4.set_xlabel('X',fontsize=16)
    ax4.set_yticks([])
    ax4.set_xlim([l1,l2])
    ax4.set_ylim([h1,h2])
#    divider = make_axes_locatable(ax4)
#    cax = divider.append_axes('right', size='1%', pad=0.05)
#    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
#    cbar4.ax.tick_params(labelsize=10)
    ax4.set_aspect(AR)      
        
    ax5 = fig.add_subplot(3,2,5)
    cp5a = ax5.tricontour(xtmp,ytmp,v,levels=lva,linewidths=0.4,colors='k',zorder=5)      
    cp5 = ax5.tricontourf(xtmp,ytmp,v,levels=lv,cmap=cm.jet,extend ='both')
    #ax5.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax5.set_title('v-CFD')
    ax5.set_xlabel('X',fontsize=16)
    ax5.set_ylabel('Y',fontsize=16)
    ax5.set_xlim([l1,l2])
    ax5.set_ylim([h1,h2])
#    divider = make_axes_locatable(ax5)
#    cax = divider.append_axes('right', size='1%', pad=0.05)
#    cbar5=plt.colorbar(cp5, cax=cax, orientation='vertical');
#    cbar5.ax.tick_params(labelsize=10)
    ax5.set_aspect(AR)
        
    ax6 = fig.add_subplot(3,2,6)
    cp6a = ax6.tricontour(xtmp,ytmp,v_pred,levels=lva,linewidths=0.4,colors='k',zorder=5)      
    cp6 = ax6.tricontourf(xtmp,ytmp,v_pred,levels=lv,cmap=cm.jet,extend ='both')
    #ax6.tricontourf(co[:,0],co[:,1],np.zeros(len(co)),colors='w')
    ax6.set_title('v-PINN')
    ax6.set_xlabel('X',fontsize=16)
    ax6.set_yticks([])
    ax6.set_xlim([l1,l2])
    ax6.set_ylim([h1,h2])
    #divider = make_axes_locatable(ax6)
    #cax = divider.append_axes('right', size='1%', pad=0.05)
    #cbar6=plt.colorbar(cp6, cax=cax, orientation='vertical');
    #cbar6.ax.tick_params(labelsize=10)
    ax6.set_aspect(AR)
    
    #fig.suptitle(" Re=40 at t=0", fontsize=20)
    
    #plt.tight_layout()
    plt.subplots_adjust( hspace = 0.05, wspace = 0.05)       
    plt.savefig('./plot/bl_%s.png'%suff,format='png',bbox_inches='tight',dpi=300)
    plt.close()


#con_plot()    

def plot_cp():
        
    #LinearNDinterpolator
    pD=np.asarray([xtmp,ytmp]).transpose()

   
    xu=np.linspace(0.1,4.9,50)
    yu=np.linspace(1e-6,1e-6,50)
        
    xl=xu
    yl=yu
   

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
    
    mei=1       
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xu,pu1,'og',linewidth=3,markevery=mei,label='CFD')
    #plt.plot(xl,pl1,'ob',linewidth=3,markevery=mei,label='CFD-lower') 
    plt.plot(xu,pu2,'r',linewidth=3,label='PINN')
    #plt.plot(xl,pl2,'k',linewidth=3,label='NN-lower')  
    plt.xlabel('X',fontsize=20)
    plt.ylabel('P',fontsize=20)
    #plt.title('Pressure Dist. over cylinder')
    plt.legend(fontsize=14)
    #plt.savefig('./plot/cp_%s.png'%suff,format='png',bbox_inches='tight', dpi=100)
    plt.show()
    
    return pu1

#plot_cp()


    
#plot
def line_plotu_sub(i):

    ######-BL thickness--######
    ########################
    
    #LinearNDinterpolator
    pD=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()
    
    aa=1.5
    xa=np.linspace(1,1,50)#not used
    ya=np.linspace(0,1,50)

    xb=np.linspace(1,1,50)
    yb=np.linspace(0,d1*aa,50)

    xc=np.linspace(2,2,50)
    yc=np.linspace(0,d2*aa,50)

    xd=np.linspace(3,3,50)
    yd=np.linspace(0,d3*aa,50)
        
    
       
        
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
    
    fac1=np.sqrt(1/(0.01*1))
    plt.figure(figsize=(6, 4), dpi=100)
    
    plt.subplot(1,3,1)
    plt.plot(bla[:,0],bla[:,1],'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(u2b,yb*fac1,'g',linewidth=3,label='PINN')
    #plt.plot(bla[:,0],bla[:,1])
    #plt.plot(u3a,ya,'r',linewidth=3,label='NN')
    #plt.legend(fontsize=14)
    plt.ylabel('$\eta$',fontsize=20)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.05,1.05)    
    
    fac1=np.sqrt(1/(0.01*2))
    plt.subplot(1,3,2)
    plt.plot(bla[:,0],bla[:,1],'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(u2c,yc*fac1,'g',linewidth=3)

    #plt.plot(u3b,yb,'r',linewidth=3)
    plt.xlabel('u/U$_\inf$',fontsize=20)
    plt.yticks([])
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.05,1.05)    
        
    fac1=np.sqrt(1/(0.01*3))
    plt.subplot(1,3,3)
    plt.plot(bla[:,0],bla[:,1],'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='Blasius')
    plt.plot(u2d,yd*fac1,'g',linewidth=3,label='CFD')

    #plt.plot(u3d,yd,'r',linewidth=3,label='NN')
    plt.legend(loc="upper left", bbox_to_anchor=[-0.02, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
    plt.yticks([])    
    #plt.xlim(-0.5,1.2)
    #plt.ylim(-0.05,1.05)    
    
    #plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
    plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
    plt.savefig('./plot/u_%s_%s.tiff'%(i,suff), format='tiff', bbox_inches='tight',dpi=100)
    #plt.ylim([-0.05,7])
    plt.show()   
    plt.close()
    
    # fp=open('./paper_files/u_%s.dat'%suff,'w')
    # fp.write('u-cfd1 y1 u-pinn1 y1 u-cfd2 y2 u-pinn2 y2  ....\n')
    # for i in range(len(u1b)):
    #     fp.write('%f %f %f %f %f %f %f %f %f %f %f %f\n'%(u1b[i],yb[i]/d1,u2b[i],yb[i]/d1,\
    #                               u1c[i],yc[i]/d2,u2c[i],yc[i]/d2,u1d[i],yd[i]/d3,u2d[i],yd[i]/d3))

    # fp.close()
    
    mei=2
    
    plt.figure(figsize=(6, 4), dpi=100)
    
    plt.subplot(1,3,1)
    plt.plot(v1b,yb/d1,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(v2b,yb/d1,'g',linewidth=3)
    #plt.plot(v3a,ya,'r',linewidth=3)
    #plt.legend(fontsize=14)
    plt.ylabel('y/$\delta$(x)',fontsize=20)
    #plt.xlim(-0.1,1.0)
    #plt.ylim(-0.05,1.05)    
    
    plt.subplot(1,3,2)
    plt.plot(v1c,yc/d2,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei)
    plt.plot(v2c,yc/d2,'g',linewidth=3)
    #plt.plot(v3b,yb,'r',linewidth=3)
    plt.xlabel('v/U$_\inf$',fontsize=20)
    plt.yticks([])
    #plt.xlim(-0.1,0.5)
    #plt.ylim(-0.05,1.05)        
    
    plt.subplot(1,3,3)
    plt.plot(v1d,yd/d3,'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=mei,label='CFD')
    plt.plot(v2d,yd/d3,'g',linewidth=3,label='PINN')
    #plt.plot(v3d,yd,'r',linewidth=3,label='NN')
    plt.yticks([])  
    plt.legend(loc="upper left", bbox_to_anchor=[0.19, 0.5], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
    #plt.ylim(-0.05,1.05)    
    #plt.xlim(-0.3,0.5)
       
    plt.figtext(0.4, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
    plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
    #plt.savefig('./plot/v_%s_%s.png'%(i,suff), format='png',bbox_inches='tight', dpi=100)
    plt.show() 
    plt.close()    
    
    # fp=open('./paper_files/v_%s.dat'%suff,'w')
    # fp.write('u-cfd1 y1 u-pinn1 y1 u-cfd2 y2 u-pinn2 y2  ....\n')
    # for i in range(len(u1b)):
    #     fp.write('%f %f %f %f %f %f %f %f %f %f %f %f\n'%(v1b[i],yb[i]/d1,v2b[i],yb[i]/d1,\
    #                               v1c[i],yc[i]/d2,v2c[i],yc[i]/d2,v1d[i],yd[i]/d3,v2d[i],yd[i]/d3))

    # fp.close()
    
    
line_plotu_sub(j)




    
    
    
    
    
    
    
    
    
    
    
    
    