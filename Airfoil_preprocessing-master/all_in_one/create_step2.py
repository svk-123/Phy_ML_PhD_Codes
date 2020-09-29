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
casedir='./picked_uiuc_aug/'
fname = [f for f in listdir(casedir) if isfile(join(casedir, f))]
fname=np.asarray(fname)
fname.sort()
np.random.seed(12453)
random.shuffle(fname)

fp=open('step2_name_0p5.txt','w')
for i in range(len(fname)):
    fp.write('%s\n'%fname[i])

fp.close()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
