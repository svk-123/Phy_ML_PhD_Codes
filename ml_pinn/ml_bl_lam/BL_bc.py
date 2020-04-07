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


plt.figure()
fp=open('./data_file/BL_wall_500.dat','w')

N=20
a=2

x=np.linspace(0,4.9,500)
y=np.linspace(0,0,500)

plt.plot(x,y)

fp.write('x y p u v\n')
for i in range(len(x)):

    fp.write('%f %f 0 1e-12 1e-12 \n'%(x[i],y[i]))

#x=np.linspace(a,a,N)
#y=np.linspace(-a,a,N)
#plt.plot(x,y)
#for i in range(len(x)):
#    fp.write('%f %f 1.0 1e-12 \n'%(x[i],y[i]))

#x=np.linspace(-a,a,N)
#y=np.linspace(-a,-a,N)
#plt.plot(x,y)
#
#for i in range(len(x)):
#    fp.write('%f %f 1.0 1e-12 \n'%(x[i],y[i]))
#
#x=np.linspace(-a,a,N)
#y=np.linspace(a,a,N)
#plt.plot(x,y)
#plt.show()
#
#for i in range(len(x)):
#    fp.write('%f %f 1.0 1e-12 \n'%(x[i],y[i]))
#fp.close()    
#    
