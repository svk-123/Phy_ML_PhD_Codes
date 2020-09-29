#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

with new parameters tanh_16_v1:

with new flow prediction network using v1.

@author: vinoth
"""
#based on parameters 

import time
start_time = time.time()


# Python 3.5
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys


from numpy import linalg as LA
import os, shutil
from skimage import io, viewer,util 
from scipy.optimize import minimize

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

from naca import naca4

path='./result_paper_v7/sp_3_tanh_max/'
name=[]
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and 'opti' in i:
        name.append(i.split('_')[1].split('.')[0])

cl=[]
cd=[]
para=[]
init_cl=[]

for i in range(len(name)):
    with open(path +'resx_%s.dat'%name[i], 'r') as infile:
        data0=infile.readlines()
       
    for j in range(len(data0)):
        if 'final-Cl' in data0[j]:
            cl.append(data0[j])
            
        if 'final-Cd' in data0[j]:
            cd.append(data0[j])            

        if 'init-Cl' in data0[j]:
            init_cl.append(data0[j])  
            
    para.append(data0[7])


for j in range(len(cl)):
    cl[j]=float(cl[j].split('=')[1])
    cd[j]=float(cd[j].split('=')[1])
    init_cl[j]=float(init_cl[j].split('=')[1])
    
    para[j]=para[j].replace('[','').replace(']','').split()
    
para=np.asarray(para)
para=para.astype(np.float)    
    
init_cl.sort()
print init_cl

print para