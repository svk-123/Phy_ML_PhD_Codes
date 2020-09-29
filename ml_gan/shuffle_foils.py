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

indir='./foil_0p5/'

fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()  
np.random.seed(123) 
idx = np.random.choice(len(fname), len(fname), replace=False)

for i in range(6000,len(fname)):
    shutil.copy('./foil_0p5/%s'%fname[idx[i]], './foil_0p5_parts/part_7/%s'%fname[idx[i]])    
    
    


