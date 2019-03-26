#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

@author: vinoth
"""

import time
start_time = time.time()

# Python 3.5
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from skimage import io, viewer,util 
from scipy import interpolate
import pandas as pd
from os import listdir
from os.path import isfile, join
import os
import shutil

import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)


path='./'

indir='./coord_CST/airfoils_9'

fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()  



'''
# check airfoil thick and remove
nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])   

coord=[]
for i in range(len(fname)):
    print ('coord',i)
    coord.append(np.loadtxt(indir+'/%s.dat'%nname[i],skiprows=1))

tmp=[]
for i in range(len(fname)):
    if (max(abs(coord[i][:,1])) >=0.245):
        tmp.append(nname[i])
        
for i in range(len(tmp)):
    os.remove(indir + '/%s.dat'%tmp[i])
'''
        


#copy files and rename
for i in range(len(fname)):
    src= indir + '/%s'%fname[i]
    dst='./coord_CST/airfoils_set_2/%05d.dat'%(i+1688)
    
    shutil.copy(src,dst)
        