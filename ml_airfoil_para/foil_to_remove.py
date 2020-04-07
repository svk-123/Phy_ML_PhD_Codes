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

indir='./coord_shifted_scaled/'
fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()
#
nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0]) 

for i in range(1000):
    print i
    coord=np.loadtxt(indir+'%s.dat'%nname[i],skiprows=1)
    
    figure=plt.figure(figsize=(6,5))
    plt0, =plt.plot(coord[:,0],coord[:,1],'k',linewidth=0.5,label='true')
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.25,0.25)    
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.savefig('./plot/%s.png'%(nname[i]), format='png',dpi=100)
    plt.close()       
