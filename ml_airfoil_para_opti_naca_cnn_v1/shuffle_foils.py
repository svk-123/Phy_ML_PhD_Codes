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

indir='./coord_naca4_for_para/'

fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()  
np.random.seed(123) 
idx = np.random.choice(len(fname), len(fname), replace=False)

for i in range(7000,len(fname)):
    shutil.copy('./coord_naca4_for_para/%s'%fname[idx[i]], './coord_naca4_for_para_parts/part3/%s'%fname[idx[i]])    
    
    


