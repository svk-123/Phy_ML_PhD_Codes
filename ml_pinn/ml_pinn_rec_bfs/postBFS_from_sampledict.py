#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:49:29 2017

This code process OF data and exports as .pkl to prepData file
for TBNN. prepData reads .pkl and process further

@author: vino
"""
# imports
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy import interpolate
from os import listdir
from os.path import isfile,isdir, join
import cPickle as pickle

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""
Re=100
tt=2591
# read data from below dir...
path='./bfs/bfs_%s_0/postProcessing/'%Re

p=np.loadtxt(path + 'sampleDict_inlet/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_inlet/%s/somePoints_U.xy'%tt)

fp=open('./data_file/Re%s/bfs_inlet.dat'%Re,'w')
fp.write('xy p u v nx ny\n')
for i in range(len(p)):
    fp.write('-1.0 %f %f %f %f %f %f\n' %(p[i,1],p[i,3],u[i,3],u[i,4], 0, 1))
fp.close()

plt.figure()
plt.plot(p[:,0],p[:,1],'o')


p=np.loadtxt(path + 'sampleDict_outlet/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_outlet/%s/somePoints_U.xy'%tt)

fp=open('./data_file/Re%s/bfs_outlet.dat'%Re,'w')
fp.write('xy p u v \n')
for i in range(len(p)):
    fp.write('7.0 %f %f %f %f %f %f\n' %(p[i,1],p[i,3],u[i,3],u[i,4], 1, 0))
fp.close()
plt.plot(p[:,0],p[:,1],'o')

p=np.loadtxt(path + 'sampleDict_upperwall/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_upperwall/%s/somePoints_U.xy'%tt)

fp=open('./data_file/Re%s/bfs_upperwall.dat'%Re,'w')
fp.write('xy p u v \n')

for i in range(49,348):
    fp.write('%f 3.0 %f 1e-12 1e-12 %f %f \n' %(p[i,0],p[i,3], 0, -1))
fp.close()

plt.plot(p[49:348,0],p[49:348,1],'o')

p=np.loadtxt(path + 'sampleDict_lowerwall/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_lowerwall/%s/somePoints_U.xy'%tt)

fp=open('./data_file/Re%s/bfs_lowerwall.dat'%Re,'w')
fp.write('xy p u v \n')

for i in range(50,431):
    fp.write('%f %f %f %f %f %f %f \n' %(p[i,0],p[i,1],p[i,3],1e-12,1e-12,-1,0))

fp.close()


plt.plot(p[50:431,0],p[50:431,1],'o')
plt.show()