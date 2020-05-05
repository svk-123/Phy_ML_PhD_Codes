#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:50:55 2020

@author: vino
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas
from os import listdir
from os.path import isfile, join

'''
##########-------------Diff. BC for re=100 wos-----#############

path='./data_file/Re100/'  
    #import wall bc
    #x,y,p,u,v
xyu_inlet=np.loadtxt(path + 'bl_inlet.dat',skiprows=1)

xyu_wall=np.loadtxt(path + 'bl_wall.dat',skiprows=1)
   
xyu_outlet_t=np.loadtxt(path + 'bl_outlet_t.dat',skiprows=1)
   
xyu_outlet_r=np.loadtxt(path + 'bl_outlet_r.dat',skiprows=1)

#sampling
xyu_s=np.loadtxt(path + 'bl_sample_x8_away.dat',skiprows=1)
  
xyu_int=np.loadtxt(path + 'bl_internal.dat',skiprows=1)    
                

######-BL thickness--######
########################

nu_=1.0/100.
x_=np.linspace(1e-12,5,100)
Rex_=x_/nu_
d_=x_/np.sqrt(Rex_)
d_=4.91*d_
############################
 
plt.figure(figsize=(16, 4), dpi=100)

plt.subplot(1,3,1)
#plt0, =plt.plot(x_s,y_s,'og',linewidth=0,ms=4,label='MSE internal pts: 54 ',zorder=8)
plt0, =plt.plot(xyu_wall[:,0:1],xyu_wall[:,1:2],'ok',linewidth=0,ms=3,label='MSE BC pts: 800',zorder=5)
plt0, =plt.plot(xyu_outlet_r[:,0:1]-5,xyu_outlet_r[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet_t[:,0:1],xyu_outlet_t[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet_r[:,0:1],xyu_outlet_r[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_int[:,0:1],xyu_int[:,1:2],'+r',linewidth=0,ms=2,label='Gov Eq. Res. pts: 20000 ',zorder=4)
plt0, =plt.plot(x_, d_,'b',lw=2, label='BL', zorder=6)
##text-1
plt.text(2.5, -0.3, "Wall: u=0", horizontalalignment='center', verticalalignment='center')
plt.text(2.5, 3.3, "Outlet: p=0", horizontalalignment='center', verticalalignment='center')
plt.text(-0.3, 1.5, "Inlet: u-specified", horizontalalignment='center', verticalalignment='center',rotation=90)
#plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
plt.xlim(-1,6)
plt.ylim(-1,4)   

plt.subplot(1,3,2)
#plt0, =plt.plot(x_s,y_s,'og',linewidth=0,ms=4,label='MSE internal pts: 54 ',zorder=8)
plt0, =plt.plot(xyu_wall[:,0:1],xyu_wall[:,1:2],'ok',linewidth=0,ms=3,label='MSE BC pts: 800',zorder=5)
plt0, =plt.plot(xyu_outlet_r[:,0:1]-5,xyu_outlet_r[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet_t[:,0:1],xyu_outlet_t[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet_r[:,0:1],xyu_outlet_r[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_int[:,0:1],xyu_int[:,1:2],'+r',linewidth=0,ms=2,label='Gov Eq. Res. pts: 20000 ',zorder=4)
plt0, =plt.plot(x_, d_,'b',lw=2, label='BL', zorder=6)
#text-2
plt.text(2.5, -0.3, "Wall: u=0,dp=0", horizontalalignment='center', verticalalignment='center')
plt.text(2.5, 3.3, "Outlet: p=0,du=0", horizontalalignment='center', verticalalignment='center')
plt.text(-0.3, 1.5, "Inlet: u-specified, dp=0", horizontalalignment='center', verticalalignment='center',rotation=90)
plt.xlabel('X',fontsize=20)
#plt.ylabel('Y',fontsize=20)
plt.xlim(-1,6)
plt.ylim(-1,4)   
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=False, shadow=False,fontsize=16)

plt.subplot(1,3,3)
#plt0, =plt.plot(x_s,y_s,'og',linewidth=0,ms=4,label='MSE internal pts: 54 ',zorder=8)
plt0, =plt.plot(xyu_wall[:,0:1],xyu_wall[:,1:2],'ok',linewidth=0,ms=3,label='MSE BC pts: 800',zorder=5)
plt0, =plt.plot(xyu_outlet_r[:,0:1]-5,xyu_outlet_r[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet_t[:,0:1],xyu_outlet_t[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet_r[:,0:1],xyu_outlet_r[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_int[:,0:1],xyu_int[:,1:2],'+r',linewidth=0,ms=2,label='Gov Eq. Res. pts: 20000 ',zorder=4)
plt0, =plt.plot(x_, d_,'b',lw=2, label='BL', zorder=6)
#text-2
plt.text(2.5, -0.3, "Wall: u=0,p-specified", horizontalalignment='center', verticalalignment='center')
plt.text(2.5, 3.3, "Outlet: p=0,u-specified", horizontalalignment='center', verticalalignment='center')
plt.text(-0.3, 1.5, "Inlet: u, p-specified", horizontalalignment='center', verticalalignment='center',rotation=90)

#plt.legend(fontsize=20)

#plt.title('%s-u'%(flist[ii]),fontsiuze=16)

plt.xlim(-1,6)
plt.ylim(-1,4)    
plt.savefig('./plot/mesh9.png', format='png',bbox_inches='tight', dpi=200)
plt.show()

'''

##########################################################
#####--------------- Diff. Sampling ---------------######


path='./data_file/Re100/'  
    #import wall bc
    #x,y,p,u,v
xyu_inlet=np.loadtxt(path + 'bl_inlet.dat',skiprows=1)

xyu_wall=np.loadtxt(path + 'bl_wall.dat',skiprows=1)
   
xyu_outlet_t=np.loadtxt(path + 'bl_outlet_t.dat',skiprows=1)
   
xyu_outlet_r=np.loadtxt(path + 'bl_outlet_r.dat',skiprows=1)

#sampling
xyu_s1=np.loadtxt(path + 'bl_sample_x8_away.dat',skiprows=1)
xyu_s2=np.loadtxt(path + 'bl_sample_x8.dat',skiprows=1)
  
xyu_int=np.loadtxt(path + 'bl_internal.dat',skiprows=1)    
                

######-BL thickness--######
########################

nu_=1.0/100.
x_=np.linspace(1e-12,5,100)
Rex_=x_/nu_
d_=x_/np.sqrt(Rex_)
d_=4.91*d_
############################
 
plt.figure(figsize=(12, 4), dpi=100)

plt.subplot(1,2,1)
plt0, =plt.plot(xyu_s1[:,0:1],xyu_s1[:,1:2],'og',linewidth=0,ms=4,label='MSE internal pts: 54 ',zorder=8)
plt0, =plt.plot(xyu_wall[:,0:1],xyu_wall[:,1:2],'ok',linewidth=0,ms=3,label='MSE BC pts: 800',zorder=5)
plt0, =plt.plot(xyu_outlet_r[:,0:1]-5,xyu_outlet_r[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet_t[:,0:1],xyu_outlet_t[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet_r[:,0:1],xyu_outlet_r[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_int[:,0:1],xyu_int[:,1:2],'+r',linewidth=0,ms=2,label='Gov Eq. Res. pts: 20000 ',zorder=4)
plt0, =plt.plot(x_, d_,'b',lw=2, label='BL', zorder=6)
##text-2
plt.text(2.5, -0.3, "Wall: u=0", horizontalalignment='center', verticalalignment='center')
plt.text(2.5, 3.3, "Outlet: p=0", horizontalalignment='center', verticalalignment='center')
plt.text(-0.3, 1.5, "Inlet: u-specified", horizontalalignment='center', verticalalignment='center',rotation=90)
#plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
plt.xlim(-1,6)
plt.ylim(-1,4)   

plt.subplot(1,2,2)
plt0, =plt.plot(xyu_s2[:,0:1],xyu_s2[:,1:2],'og',linewidth=0,ms=4,label='MSE internal pts: 54 ',zorder=8)
plt0, =plt.plot(xyu_wall[:,0:1],xyu_wall[:,1:2],'ok',linewidth=0,ms=3,label='MSE BC pts: 800',zorder=5)
plt0, =plt.plot(xyu_outlet_r[:,0:1]-5,xyu_outlet_r[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet_t[:,0:1],xyu_outlet_t[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_outlet_r[:,0:1],xyu_outlet_r[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
plt0, =plt.plot(xyu_int[:,0:1],xyu_int[:,1:2],'+r',linewidth=0,ms=2,label='Gov Eq. Res. pts: 20000 ',zorder=4)
plt0, =plt.plot(x_, d_,'b',lw=2, label='BL', zorder=6)
##text-2
plt.text(2.5, -0.3, "Wall: u=0", horizontalalignment='center', verticalalignment='center')
plt.text(2.5, 3.3, "Outlet: p=0", horizontalalignment='center', verticalalignment='center')
plt.text(-0.3, 1.5, "Inlet: u-specified", horizontalalignment='center', verticalalignment='center',rotation=90)
#plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
plt.xlim(-1,6)
plt.ylim(-1,4) 

#plt.legend(fontsize=20)

#plt.title('%s-u'%(flist[ii]),fontsiuze=16)
plt.legend(loc='upper center', bbox_to_anchor=(0.0, 1.3), ncol=4, fancybox=False, shadow=False,fontsize=16)
plt.xlim(-1,6)
plt.ylim(-1,4)    
plt.savefig('./plot/mesh9.png', format='png',bbox_inches='tight', dpi=200)
plt.show()