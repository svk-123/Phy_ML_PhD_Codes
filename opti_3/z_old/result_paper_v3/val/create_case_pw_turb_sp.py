#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 01:02:17 2019

@author: vino
"""

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
    
    #line no to be modified (as given in gedit from 1...)
    l1=20
    l2=32
    for j in range(l1-1):
        fp.write('%s'%data0[j])

    for j in range(l1-1,l1):
        fp.write('internalField   uniform (%0.6f %0.6f 0); \n'%(u,v))
  
    for j in range(l1,l2-1):
        fp.write('%s'%data0[j])
        
    for j in range(l2-1,l2):
        fp.write('        value           uniform (%0.6f %0.6f 0); \n'%(u,v))
      
    for j in range(l2,len(data0)):
        fp.write('%s'%data0[j])       

    fp.close()
    
    l1=20
    #change nu for Re.
    with open(name + '/constant/transportProperties', 'r') as infile:
        data0=infile.readlines()
        
    fp= open(name + '/constant/transportProperties', "w+")
    
    for j in range(l1-1):
        fp.write('%s'%data0[j])

    for j in range(l1-1,l1):
        fp.write('nu              [0 2 -1 0 0 0 0] %0.8f ; \n'%(nu))
        
    for j in range(l1,len(data0)):
        fp.write('%s'%data0[j])       

    fp.close()


    l1=22
    l2=23
    #change forcecoeff Dir.
    with open(name + '/system/forceCoeffs', 'r') as infile:
        data0=infile.readlines()
        
    fp= open(name + '/system/forceCoeffs', "w+")
    
    for j in range(l1-1):
        fp.write('%s'%data0[j])

    for j in range(l1-1,l1):
        fp.write('	    liftDir             (%f %f 0); \n'%(-v,u))
        
    for j in range(l1,l2):
        fp.write('	    dragDir             (%f %f 0); \n'%(u, v))     
        
    for j in range(l2,len(data0)):
        fp.write('%s'%data0[j])       

    fp.close()


    l1=19
    #change nutilda
    with open(name + '/0/nuTilda', 'r') as infile:
        data0=infile.readlines()
        
    fp= open(name + '/0/nuTilda', "w+")
    
    for j in range(l1-1):
        fp.write('%s'%data0[j])

    for j in range(l1-1,l1):
        fp.write('internalField   uniform %f; \n'%(nu*10))
        
    for j in range(l1,len(data0)):
        fp.write('%s'%data0[j])       

    fp.close()


    l1=19
    #change nutilda
    with open(name + '/0/nut', 'r') as infile:
        data0=infile.readlines()
        
    fp= open(name + '/0/nut', "w+")
    
    for j in range(l1-1):
        fp.write('%s'%data0[j])

    for j in range(l1-1,l1):
        fp.write('internalField   uniform %f; \n'%(nu))
        
    for j in range(l1,len(data0)):
        fp.write('%s'%data0[j])       

    fp.close()

#------ path & create files---------------------------------------------#
mshdir='./sp_3_tanh_max/mesh'
#fname = [f for f in listdir(mshdir) if isdir(join(mshdir, f))]
#fname.sort()

dst1='./sp_3_tanh_max/foam'

relist=np.asarray([80000])
aoalist=np.asarray([8])

#relist=np.asarray([10000])
#aoalist=np.asarray([1,5])

#copy case
for i in range(1):
    
    print i
    
    np.random.shuffle(relist)
    
    dst2= dst1
        
    if not os.path.exists(dst2):
        os.makedirs(dst2)
            
    for reno in range(1):
        
        np.random.shuffle(aoalist)
            
        for aoa in range(1):
            
            src='./case_temp/'
            dst3=dst2 + '/%07d_%02d'%(relist[reno],aoalist[aoa])
    
            if os.path.exists(dst3):
                shutil.rmtree(dst3)
        
            shutil.copytree(src,dst3) 
            shutil.copytree(mshdir, dst3 + '/constant/polyMesh')  
        
            change_re_aoa(dst3,relist[reno],aoalist[aoa])
            



