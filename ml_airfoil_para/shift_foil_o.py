#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:42:22 2019

@author: vino
"""

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

import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)


path='./'

indir='./coord_seligFmt_formatted'

fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()  


nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])   

coord=[]
for i in range(len(fname)):
    #print i
    coord.append(np.loadtxt(indir+'/%s.dat'%nname[i],skiprows=1))

xx=np.loadtxt('xx.txt')
foil_fp=[]
count=0

fp=open('foil_shift_details.txt','w+')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

for i in range(len(coord)):
    print i
    idx=find_nearest(coord[i][:,0],0)
    if(coord[i][idx,1]!=0):
        fp.write("%s    %f\n"%(nname[i],coord[i][idx,1]))
        coord[i][:,1]=coord[i][:,1]-coord[i][idx,1]
        
    
fp.close()    
    
    
    
    


