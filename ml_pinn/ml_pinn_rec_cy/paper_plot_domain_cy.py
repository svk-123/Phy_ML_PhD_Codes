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
path='./data_file/'  
#import wall bc
#x,y,p,u,v
xyu_inlet=np.loadtxt(path + 'cy_inlet3_3333.dat',skiprows=1)

xyu_wall=np.loadtxt(path + 'cy_wall_200.dat',skiprows=1)

xyu_outlet_t=np.loadtxt(path + 'cy_outlet1_3333.dat',skiprows=1)
#sampling
xyu_s=np.loadtxt(path + 'cy_sample_wake_x5_10.dat',skiprows=1)

xyu_int=np.loadtxt(path + 'cy_internal_3333.dat',skiprows=1)    

        



############################
 
plt.figure(figsize=(6, 4), dpi=100)
plt0, =plt.plot(xyu_wall[:,0:1],xyu_wall[:,1:2],'ok',linewidth=0,ms=3,label='MSE BC pts: 200',zorder=5)
plt0, =plt.plot(xyu_outlet_t[:,0:1],xyu_outlet_t[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_inlet[:,0:1],xyu_inlet[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_int[:,0:1],xyu_int[:,1:2],'+r',linewidth=0,ms=2,label='Gov Eq. Res. pts: 10000 ',zorder=4)
plt0, =plt.plot(xyu_s[:,0:1],xyu_s[:,1:2],'og',linewidth=0,ms=4,label='MSE internal pts: 50 ',zorder=8)

#
###text-1
#plt.text(2.5, -0.3, "Wall: u=0", horizontalalignment='center', verticalalignment='center')
#plt.text(2.5, 3.3, "Outlet: p=0", horizontalalignment='center', verticalalignment='center')
#plt.text(-0.3, 1.5, "Inlet: u-specified", horizontalalignment='center', verticalalignment='center',rotation=90)

#plt.legend(fontsize=20)
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
#plt.title('%s-u'%(flist[ii]),fontsiuze=16)
plt.legend(loc='upper center', bbox_to_anchor=(1.45, 1), ncol=1, fancybox=False, shadow=False,fontsize=16)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.savefig('./plot/mesh9.png', format='png',bbox_inches='tight', dpi=200)
plt.show()
