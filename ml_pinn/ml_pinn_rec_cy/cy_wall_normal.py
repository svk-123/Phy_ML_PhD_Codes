#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 02:28:22 2019

@author: vino
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import pickle

theta=np.linspace(0,360,101)*np.pi/180.

x=0.5*np.cos(theta)
y=0.5*np.sin(theta)


xc=np.zeros(100)   
yc=np.zeros(100)  

nx=np.zeros(100)
ny=np.zeros(100)
    
for i in range(100):
    xc[i]=0.5*(x[i]+x[i+1])
    yc[i]=0.5*(y[i]+y[i+1])   
    dx=x[i+1]-x[i]
    dy=y[i+1]-y[i]
    ds=np.sqrt( dx**2 + dy**2 )
    nx[i]=dy/ds
    ny[i]=-dx/ds
    #dp_n=dp_x*nx+dp_y*ny


fp=open('cy_wall_bc_nxny_100.dat','w')
fp.write('x y u v nx ny: use pn=nx*nx+py*ny')
for i in range(100):
    fp.write('%f %f 1e-12 1e-12  %f %f \n'%(xc[i],yc[i],nx[i],ny[i]))
        
fp.close()    
    
plt.figure()
plt.plot(x,y)
plt.show()

