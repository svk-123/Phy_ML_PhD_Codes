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
from scipy.interpolate import griddata

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 



Re=100
suff='%s_ws_pinn'%Re    
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
    
    path1='./tf_model/case_1_Re100_nodp_nodv_ws_9x5/tf_model/'
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

   
   

#plot
def line_plot1():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(u1a,ya,'-og',linewidth=3,label='CFD')
    plt0, =plt.plot(u2a,ya,'r',linewidth=3,label='PINN')
    plt.legend(fontsize=20)
    plt.xlabel('u-velocity',fontsize=20)
    plt.ylabel('Y',fontsize=20)
    #plt.title('%s-u'%(flist[ii]),fontsiuze=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-u.png'%(suff), format='png',bbox_inches='tight', dpi=100)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(xb,v1a,'-og',linewidth=3,label='CFD')
    plt0, =plt.plot(xb,v2a,'r',linewidth=3,label='PINN')    
    plt.legend(fontsize=20)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('v-velocity' ,fontsize=20)
    #plt.title('%s-v'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-v.png'%(suff), format='png',bbox_inches='tight', dpi=100)
    plt.show()     
    
#plot
def plot(xp,yp,zp,nc,name):

    plt.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = plt.tricontour(xp,yp,zp,nc,linewidths=0.3,colors='k',zorder=5)
    cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet,zorder=0)
   # v= np.linspace(0, 0.05, 15, endpoint=True)
    #cp = plt.tricontourf(xp,yp,zp,v,cmap=cm.jet,extend='both')
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    plt.colorbar()
    #plt.title('%s  '%flist[ii]+name)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    plt.savefig('./plot/%s.png'%(suff+name), format='png',bbox_inches='tight', dpi=100)
    plt.show()
          
plot(xtmp,ytmp,u,20,'u-cfd')
plot(xtmp,ytmp,tout[:,0],20,'u-nn')
plot(xtmp,ytmp,abs(u-tout[:,0]),20,'u-error')
    
plot(xtmp,ytmp,v,20,'v-cfd')
plot(xtmp,ytmp,tout[:,1],20,'v-nn')
plot(xtmp,ytmp,abs(v-tout[:,1]),20,'v-error')


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




def stream_plot(u11,v11,name):
    
    fig = plt.figure(figsize=(6, 6),dpi=100)
    
    pts_x=np.linspace(0.01,0.99,200)
    pts_y=np.linspace(0.01,0.99,200)
    xx,yy=np.meshgrid(pts_x,pts_y)
    
    pts=np.concatenate((xx.flatten()[:,None],yy.flatten()[:,None]),axis=1)
    
    points=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()
    grid_y, grid_x = np.mgrid[0:1:200j, 0:1:200j]
    
    u1 = griddata(points, u11, (grid_x, grid_y), method='linear')
    v1 = griddata(points, v11, (grid_x, grid_y), method='linear')

    ax1 = fig.add_subplot(1,1,1)
    ax1.streamplot(grid_x,grid_y,u1,v1,density=4,linewidth=0.4,color='k', cmap=cm.jet,arrowsize=0.02,\
                   minlength=0.1, maxlength=4.0, zorder=0)
#    seed_points=np.array([xx.flatten(), yy.flatten()])
#    ax1.streamplot(grid_x,grid_y,u1,v1,density=4,linewidth=1,color='k', cmap=cm.jet,arrowsize=0.02,\
#                   minlength=0.1, start_points=seed_points.T, maxlength=4.0, zorder=0)    
    
        
    #ax1.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='gray',zorder=5)
    #ax1.set_title('%s'%name)
    ax1.set_xlabel('X',fontsize=20)
    ax1.set_ylabel('Y',fontsize=20)
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    #plt.subplots_adjust( wspace=0.2,hspace=0.3)
    ax1.set_aspect(1)
     
         
    
    
    plt.subplots_adjust(top = 1.2, bottom = 0.1, right = 0.98, left = 0.05, hspace = 0.0, wspace = 0.25)   
    plt.savefig('./plot/stream_%s.png'%(name),format='png',bbox_inches='tight',dpi=200)
    plt.show()
    plt.close()

stream_plot(u_pred,v_pred,'ws_pinn')







    
    
    
    
    
    
    
    
    
    
    
    
    
