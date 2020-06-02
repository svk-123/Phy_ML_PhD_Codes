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

# read data from below dir...
path='./data_file/'


fp=open( path + 'naca4412_internal_combined_1211.dat','w')
fp.write('x y p u v\n')

xy=np.loadtxt(path + 'naca4412_internal_1211.dat',skiprows=1)
for i in range(len(xy)):
    fp.write('%f %f %f %f %f %f %f\n' %(xy[i,0],xy[i,1],xy[i,2],xy[i,3],xy[i,4], 0, 1))
    
xy=np.loadtxt(path + 'naca4412_inlet3_1211.dat',skiprows=1)
for i in range(len(xy)):
    fp.write('%f %f %f %f %f %f %f\n' %(xy[i,0],xy[i,1],xy[i,2],xy[i,3],xy[i,4], 0, 1))    
        
xy=np.loadtxt(path + 'naca4412_outlet1_1211.dat',skiprows=1)
for i in range(len(xy)):
    fp.write('%f %f %f %f %f %f %f\n' %(xy[i,0],xy[i,1],xy[i,2],xy[i,3],xy[i,4], 0, 1))

xy=np.loadtxt(path + 'naca4412_wall_300.dat',skiprows=1)
for i in range(len(xy)):
    fp.write('%f %f %f %f %f %f %f\n' %(xy[i,0],xy[i,1],xy[i,2],xy[i,3],xy[i,4], 0, 1))    
    
fp.close()
