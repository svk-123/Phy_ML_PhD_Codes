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
Re=1000
#100-3917
#1000-2987
tt=2987
# read data from below dir...
path='./ldc/ldc_%s_0/postProcessing/'%Re

p=np.loadtxt(path + 'sampleDict_fwall/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_fwall/%s/somePoints_U.xy'%tt)

fp=open('./data_file/Re%s/ldc_fwall.dat'%Re,'w')
fp.write('xy p u v nx ny\n')
for i in range(len(p)):
    fp.write('%f %f %f %f %f %f %f\n' %(p[i,0],p[i,1],p[i,3],u[i,3],u[i,4], 0, 1))
fp.close()

p=np.loadtxt(path + 'sampleDict_mwall/%s/somePoints_p.xy'%tt)
u=np.loadtxt(path + 'sampleDict_mwall/%s/somePoints_U.xy'%tt)

fp=open('./data_file/Re%s/ldc_mwall.dat'%Re,'w')
fp.write('xy p u v nx ny\n')
for i in range(len(p)):
    fp.write('%f %f %f %f %f %f %f\n' %(p[i,0],p[i,1],p[i,3],u[i,3],u[i,4], 0, 1))
fp.close()