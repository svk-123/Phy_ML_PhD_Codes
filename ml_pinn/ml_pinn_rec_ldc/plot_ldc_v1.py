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
suff='%s_wc_wt'%Re    
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
    
    path1='./tf_model/case_1_Re%s_ws_wc_wt1000/tf_model/'%Re
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











    
    
    
    
    
    
    
    
    
    
    
    
    
