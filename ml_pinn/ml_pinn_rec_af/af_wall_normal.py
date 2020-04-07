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

#load datafile
xy=np.loadtxt('./data_file/naca0012_200_cos.dat')
x=xy[:,0]
y=xy[:,1]

xc=np.zeros(len(xy)-1)   
yc=np.zeros(len(xy)-1)   
nx=np.zeros(len(xy)-1)   
ny=np.zeros(len(xy)-1)   
    
for i in range(len(xy)-1):
    xc[i]=0.5*(x[i]+x[i+1])
    yc[i]=0.5*(y[i]+y[i+1])   
    dx=x[i+1]-x[i]
    dy=y[i+1]-y[i]
    ds=np.sqrt( dx**2 + dy**2 )
    nx[i]=dy/ds
    ny[i]=-dx/ds
    #dp_n=dp_x*nx+dp_y*ny


fp=open('af_wall.dat','w')
fp.write('x y p u v nx ny: use pn=nx*nx+py*ny')
for i in range(len(xy)-1):
    fp.write('%f %f 0.0 1e-12 1e-12  %f %f \n'%(xc[i],yc[i],nx[i],ny[i]))
        
fp.close()    
    
plt.figure()
plt.plot(x,y)
plt.show()

plt.figure()
plt.plot(xc,nx,xc,ny)
plt.show()