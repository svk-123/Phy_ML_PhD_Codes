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



name='naca0012_Re200_30'


fp=open('./data_file/naca0012_wall_300.dat','w')
fp.write('x y t p u v nx ny\n')

t1=24.0
t2=25.0
          
tt = np.linspace(t1,t2,int (round((t2-t1)/0.05)+1) )
         
    
mytt = tt-t1
mytt = mytt


for kk in range(len(tt)):
    
    if(kk==0 or kk==20):
       
        # read data from below dir...
        path='./foil/training/%s/postProcessing/'%name
        
        p=np.loadtxt(path + 'sampleDict_wall/%s/somePoints_p.xy'%int(tt[kk]))
        u=np.loadtxt(path + 'sampleDict_wall/%s/somePoints_U.xy'%int(tt[kk]))
        
    else:
       
        # read data from below dir...
        path='./foil/training/%s/postProcessing/'%name
        
        p=np.loadtxt(path + 'sampleDict_wall/%s/somePoints_p.xy'%tt[kk])
        u=np.loadtxt(path + 'sampleDict_wall/%s/somePoints_U.xy'%tt[kk])    
        
    for i in range(len(p)):
        fp.write('%f %f %f %f %f %f %f %f\n' %(p[i,0],p[i,1],mytt[kk],p[i,3],1e-12,1e-12, 0, 1))
    
fp.close()

plt.figure()
plt.plot(p[:,0],p[:,1],'o')
plt.show()







