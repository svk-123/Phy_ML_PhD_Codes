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
import pandas as pd
from scipy import interpolate
from os import listdir
from os.path import isfile, isdir, join
from numpy import random

#load file name
casedir='./cst/'
fname = [f for f in listdir(casedir) if isfile(join(casedir, f))]
fname=np.asarray(fname)
fname.sort()


for i in range(len(fname)):
    xy=np.loadtxt(casedir + '%s'%fname[i],skiprows=1)
    
    fp=open('./cst_reformat/cst%s'%fname[i],'w')
    fp.write('%f %f \n'%(xy[0,0],xy[0,1]))
    for j in range(len(xy)):
        fp.write('%f %f \n'%(xy[::-1][j,0],xy[::-1][j,1]))
    fp.close()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
