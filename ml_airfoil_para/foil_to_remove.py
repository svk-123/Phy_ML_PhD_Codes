#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:58:37 2019

@author: vino
"""
# imports
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import interpolate 
import math
from os import listdir
from os.path import isfile,isdir, join
import shutil

indir='./coord_shifted_scaled'
fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()
#
nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0]) 

#coord=[]
#for i in range(len(fname)):
#    print i
#    coord.append(np.loadtxt(indir+'/%s.dat'%nname[i],skiprows=1))
    
    
#fp=open('foil1.dat','w+')
#for i in range(len(fname)):
#    if( coord[i][:,1].max() > 0.24):
#        fp.write('%s \n'%nname[i])
#fp.close()        
#with open('./foil1.dat', 'r') as infile:
#   tmp=infile.readlines()
#for i in range(len(tmp)):
#   os.remove('./coord_shifted_scaled/%s.dat'%tmp[i].strip())
    
fp=open('foil2.dat','w+')    
for ttt in range(1):    
    for i in range(100):
        print i
        l=len(coord[i])
        ind=np.argmin(coord[i][:,0])
        
        up_x=coord[i][:ind+1,0]
        up_y=coord[i][:ind+1,1]
        
        lr_x=coord[i][ind:,0]
        lr_y=coord[i][ind:,1]    
        lr_y=lr_y[::-1]
        for j in range(len(up_y)):
            if(lr_y[j] > up_y[j]):
                fp.write('%s \n'%nname[i])
        
fp.close()        
