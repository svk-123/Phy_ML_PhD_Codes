#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 02:46:10 2020

@author: vino
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas
from os import listdir
from os.path import isfile, join



#############################################################
path='./data_file/Re100/'  
#import wall bc
#x,y,p,u,v
xyu_inlet=np.loadtxt(path + 'bfs_inlet.dat',skiprows=1)

xyu_outlet=np.loadtxt(path + 'bfs_outlet.dat',skiprows=1)

xyu_upperwall=np.loadtxt(path + 'bfs_upperwall.dat',skiprows=1)

xyu_lowerwall=np.loadtxt(path + 'bfs_lowerwall.dat',skiprows=1)

#sampling
xyu_s=np.loadtxt(path + 'bfs_sample_x5_30_f3.dat',skiprows=1)
 
xyu_int_=np.loadtxt(path + 'bfs_internal.dat',skiprows=1)
#xyu_int=49753
idx2 = np.random.choice(len(xyu_int_), 30000, replace=False)
xyu_int=xyu_int_[idx2,:]



##### 
plt.figure(figsize=(6, 3), dpi=100)
plt0, =plt.plot(xyu_s[:,0:1],xyu_s[:,1:2],'og',linewidth=0,ms=4,label='MSE internal pts: 50 ',zorder=8)
plt0, =plt.plot(xyu_upperwall[:,0:1],xyu_upperwall[:,1:2],'ok',linewidth=0,ms=3,label='MSE BC pts: 800',zorder=5)
plt0, =plt.plot(xyu_lowerwall[:,0:1],xyu_lowerwall[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet[:,0:1],xyu_outlet[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_inlet[:,0:1],xyu_inlet[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_int[:,0:1],xyu_int[:,1:2],'+r',linewidth=0,ms=2,label='Gov Eq. Res. pts: 15000 ',zorder=4)

###text-1
#plt.text(2.5, -0.3, "Wall: u=0", horizontalalignment='center', verticalalignment='center')
#plt.text(2.5, 3.3, "Outlet: p=0", horizontalalignment='center', verticalalignment='center')
#plt.text(-0.3, 1.5, "Inlet: u-specified", horizontalalignment='center', verticalalignment='center',rotation=90)

#plt.legend(fontsize=20)
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
#plt.title('%s-u'%(flist[ii]),fontsiuze=16)
plt.legend(loc='upper center', bbox_to_anchor=(1.45, 1), ncol=1, frameon=False, fancybox=False, shadow=False,fontsize=16)
plt.xlim(-1.1,7.1)
plt.ylim(-0.1,3.1)    
plt.savefig('./plot/mesh1.tiff', format='tiff',bbox_inches='tight', dpi=200)
plt.show()

