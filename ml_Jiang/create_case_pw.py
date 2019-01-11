#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:12:02 2017

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

#.........  Fn to change Re,U in case file ----------#

def change_re_aoa(name,myre,myaoa):

    print name
    nu=1.0/float(myre)
    u=np.cos(np.radians(myaoa))
    v=np.sin(np.radians(myaoa))
    
    #change U for aoa.
    with open(name + '/0/U', 'r') as infile:
        data0=infile.readlines()
        
    fp= open(name + '/0/U', "w+")
    
    for j in range(18):
        fp.write('%s'%data0[j])

    for j in range(18,19):
        fp.write('internalField   uniform (%0.6f %0.6f 0); \n'%(u,v))
  
    for j in range(20,26):
        fp.write('%s'%data0[j])
        
    for j in range(26,27):
        fp.write('        freestreamValue uniform (%0.6f %0.6f 0); \n'%(u,v))
      
    for j in range(28,len(data0)):
        fp.write('%s'%data0[j])       

    fp.close()
    
    
    #change nu for Re.
    with open(name + '/constant/transportProperties', 'r') as infile:
        data0=infile.readlines()
        
    fp= open(name + '/constant/transportProperties', "w+")
    
    for j in range(21):
        fp.write('%s'%data0[j])

    for j in range(21,22):
        fp.write('nu              [0 2 -1 0 0 0 0] %0.8f ; \n'%(nu))
        
    for j in range(22,len(data0)):
        fp.write('%s'%data0[j])       

    fp.close()


#------ path & create files---------------------------------------------#
mshdir='./mesh'
fname = [f for f in listdir(mshdir) if isdir(join(mshdir, f))]
fname.sort()

dst1='./foam_run'

#relist=np.asarray([100,200,300,400,600,800,1000,1200,1500,1800,2000])
#aoalist=np.asarray([0,2,4,6,8,10,12,14])

relist=np.asarray([500,1000])
aoalist=np.asarray([1,5])

#copy case
for i in range(len(fname)):
    
    print i
    
    np.random.shuffle(relist)
    
    dst2= dst1+'/%s'%fname[i]
        
    if not os.path.exists(dst2):
        os.makedirs(dst2)
            
    for reno in range(2): # change no ...
        
        np.random.shuffle(aoalist)
            
        for aoa in range(2): # change no ...
            
            src='./template_case_lam/'
            dst3=dst2 + '/%s_%07d_%02d'%(fname[i],relist[reno],aoalist[aoa])
    
            if os.path.exists(dst3):
                shutil.rmtree(dst3)
        
            shutil.copytree(src,dst3) 
            shutil.copytree(mshdir + '/%s'%fname[i], dst3 + '/constant/polyMesh')  
        
            change_re_aoa(dst3,relist[reno],aoalist[aoa])
            



