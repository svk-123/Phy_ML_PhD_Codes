#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:56:16 2019
@author: vino
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas
from os import listdir
from os.path import isfile, join

N=50

b=5

theta=np.linspace(0,360,200)
theta=theta*np.pi/180.0

x=5*np.cos(theta)
y=5*np.sin(theta)

u=5*np.cos(14*np.pi/180)
v=5*np.sin(14*np.pi/180)

plt.figure()
plt.plot(x,y,'o')
plt.show()

fp=open('./data_file/af_inout_xy_200.dat','w')

for i in range(len(x)):
    fp.write("%f %f \n"%(x[i],y[i]))

fp.close()

###----wall bc-----#########
xy=np.loadtxt('./data_file/naca4518.dat')

    
idx = np.random.choice(len(xy), 100, replace=False)
xy=xy[idx,:]

fp=open('./data_file/af_wall_bc_100.dat','w')

for i in range(len(xy)):
    fp.write('%f %f 1e-12 1e-12 \n'%(xy[i,0],xy[i,1]))

fp.close()

plt.figure()
plt.plot(xy[:,0],xy[:,1],'o')
plt.show()
