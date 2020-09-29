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

import cPickle as pickle
import pandas
from skimage import io, viewer,util 
import shutil
np.set_printoptions(threshold=np.inf)

path='./'

indir='./picked_foil_0p5_parts/part_7/'
outdir='./xflr5_foils/part_7/'
fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()  

nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])
nname=np.asarray(nname)    

for i in range(len(nname)):    
    fp=open(outdir + '%s.dat'%nname[i],'w')
    fp.write('%s\n'%nname[i])
    xy=np.loadtxt(indir + '%s.dat'%nname[i])
    for j in range(len(xy)):
        fp.write('%f %f\n'%(xy[j,0],xy[j,1]))
    fp.close()    
    


