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

#load file name
casedir='./case'
fname1 = [f for f in listdir(casedir) if isdir(join(casedir, f))]
fname1.sort()

print 'writing..'
fp= open('./case/prun.sh',"w+")
        
fp.write('#!/bin/sh \n\
# Source tutorial run functions \n\
. $WM_PROJECT_DIR/bin/tools/RunFunctions \n') 
fp.write('caseList=\"')

for i in range(len(fname1)):
        
    dir1=casedir + '/%s'%fname1[i]
    fname2 = [f for f in listdir(dir1) if isdir(join(dir1, f))]


    for j in range(len(fname2)):
    
        fp.write('%s/%s '%(fname1[i],fname2[j]))
        
fp.write('\" \n')        
fp.write('for case in $caseList \n\
         do \n\
             cd $case \n\
                     \n\
                     decomposePar \n\
                     mpirun -np 8 simpleFoam -parallel > log.simpleFoam \n\
                     reconstructPar \n\
                     rm -r processor* \n\
                     writeCellCentres  \n\
                     cd .. \n\
         done ')
           
fp.close()    
    
    
    