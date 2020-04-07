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





x1=np.linspace(-0.5,-0.5,20)
y1=np.linspace(-0.5,0.5,20)
x2=np.linspace(-0.5,0.5,20)
y2=np.linspace(0.5,0.5,20)
x3=np.linspace(0.5,0.5,20)
y3=np.linspace(-0.5,0.5,20)
x4=np.linspace(-0.5,0.5,20)
y4=np.linspace(-0.5,-0.5,20)

x=np.concatenate((x1,x2,x3,x4),axis=0)
y=np.concatenate((y1,y2,y3,y4),axis=0)


plt.figure()
plt.plot(x1,y1,'o')
plt.plot(x2,y2,'o')
plt.plot(x3,y3,'o')
plt.plot(x4,y4,'o')
plt.show()


fp=open('cy_wall_bc_80.dat','w')

fp.write('x y t p u v \n')
for j in range(1):
    for i in range(len(x)):
        fp.write('%f %f 0.00 1e-12 1e-12 \n'%(x[i],y[i]))
        
fp.close()
    
