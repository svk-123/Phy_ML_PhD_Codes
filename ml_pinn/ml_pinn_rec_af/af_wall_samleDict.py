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



name='naca4412_000100_20'
tt=327

# read data from below dir...
path='./foil/%s/postProcessing/'%name

p=np.loadtxt(path + 'sampleDict_wall/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_wall/%s/somePoints_U.xy'%tt)

fp=open('./data_file/naca4412_wall_300.dat','w')
fp.write('xy p u v nx ny\n')
for i in range(len(p)):
    fp.write('%f %f %f %f %f %f %f\n' %(p[i,0],p[i,1],p[i,3],1e-12,1e-12, 0, 1))
fp.close()

plt.figure()
plt.plot(p[:,0],p[:,1],'o')
plt.show()







