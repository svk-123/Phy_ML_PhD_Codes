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
Re=40000
tt=153
# read data from below dir...
path='./bl/bl_%s_0/postProcessing/'%Re

p=np.loadtxt(path + 'sampleDict_out_t/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_out_t/%s/somePoints_U.xy'%tt)

fp=open('./data_file/Re%s/bl_outlet_t.dat'%Re,'w')
fp.write('xy p u v nx ny\n')
for i in range(len(p)):
    fp.write('%f 3.0 %f %f %f %f %f\n' %(p[i,0],p[i,3],u[i,3],u[i,4], 0, 1))
fp.close()
   
p=np.loadtxt(path + 'sampleDict_out_r/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_out_r/%s/somePoints_U.xy'%tt)

fp=open('./data_file/Re%s/bl_outlet_r.dat'%Re,'w')
fp.write('xy p u v \n')
for i in range(len(p)):
    fp.write('5.0 %f %f %f %f %f %f\n' %(p[i,1],p[i,3],u[i,3],u[i,4], 1, 0))
fp.close()


p=np.loadtxt(path + 'sampleDict_wall/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_wall/%s/somePoints_U.xy'%tt)

fp=open('./data_file/Re%s/bl_wall.dat'%Re,'w')
fp.write('xy p u v \n')
for i in range(len(p)):
    fp.write('%f 0.0 %f 1e-12 1e-12 %f %f \n' %(p[i,0],p[i,3], 0, -1))
fp.close()

p=np.loadtxt(path + 'sampleDict_inlet/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_inlet/%s/somePoints_U.xy'%tt)

fp=open('./data_file/Re%s/bl_inlet.dat'%Re,'w')
fp.write('xy p u v \n')
for i in range(len(p)):
    fp.write('0.0 %f %f %f %f %f %f \n' %(p[i,1],p[i,3],u[i,3],u[i,4],-1,0))
fp.close()