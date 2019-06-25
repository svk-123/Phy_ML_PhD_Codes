#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

This code to make row of 10 images randomly 
for testing 3D parametrization
- Intiial check

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
np.set_printoptions(threshold=np.inf)


#load
path='./data_file/'
data_file='foil_param_216_no_aug.pkl'

inp=[]
out=[]
xx=[]
name=[]

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
print result[-1:]    
inp.extend(result[0])
out.extend(result[1])
xx.extend(result[2])
name.extend(result[3])

inp=np.asarray(inp)
out=np.asarray(out)

np.random.seed(123)

new_inp=[]
new_out=[]

for i in range(500):
    I=np.random.randint(0,len(inp),10)
    
    inp_tmp=inp[I]
    out_tmp=out[I]
   
    new_inp.append(inp_tmp.transpose())
    new_out.append(out_tmp.flatten())
        
new_inp=np.asarray(new_inp)
new_out=np.asarray(new_out)    
    
info='[inp,out,xx,info]'    
data2=[new_inp,new_out,xx,info]
with open(path+'foil_param_3d_1.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)

    
    
    

