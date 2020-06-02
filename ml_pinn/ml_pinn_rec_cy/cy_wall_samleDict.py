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

'''
theta=np.linspace(0,360,100)*np.pi/180.

x=0.5*np.cos(theta)
y=0.5*np.sin(theta)

fp=open('cy_wall_bc_100.dat','w')

for i in range(len(x)):
    fp.write('%f %f 1e-12 1e-12 \n'%(x[i],y[i]))
    
fp.close()    
'''    

Re=40
tt=321
# read data from below dir...
path='./cy/cy_%s_0/postProcessing/'%Re

p=np.loadtxt(path + 'sampleDict_wall/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_wall/%s/somePoints_U.xy'%tt)

fp=open('./data_file/cy_wall.dat','w')
fp.write('xy p u v nx ny\n')
for i in range(len(p)):
    fp.write('%f %f %f %f %f %f %f\n' %(p[i,0],p[i,1],p[i,3],1e-12,1e-12, 0, 1))
fp.close()

plt.figure()
plt.plot(p[:,0],p[:,1],'o')
plt.show()







