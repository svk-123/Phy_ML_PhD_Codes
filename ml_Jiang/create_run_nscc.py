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


print 'writing..'
fp= open('./run_jiang',"w+")

fp.write('#!/bin/bash \n')
fp.write('#PBS -N cases \n')         
fp.write('#PBS -P Personal \n')         
fp.write('#PBS -l select=1:ncpus=8:mpiprocs=8:mem=8GB \n')         
fp.write('#PBS -l place=free \n')         
fp.write('#PBS -l walltime=24:00:00\n') 
fp.write('#PBS -j oe \n') 
fp.write('module load openfoam5/5.0 \n\n\n')


#load file name
path='./'
dirname='foam_run'

casedir=path+dirname
fname1 = [f for f in listdir(casedir) if isdir(join(casedir, f))]
fname1.sort()

for i in range(len(fname1)):
    
    dir1=casedir + '/%s'%fname1[i]
    fname2 = [f for f in listdir(dir1) if isdir(join(dir1, f))]
    
    for j in range(len(fname2)):
     
    
        fp.write('cd /home/users/nus/e0021524/scratch/foam_run/%s/%s/%s \n\
        decomposePar -force	\n\
        mpiexec simpleFoam -parallel >log \n\
        reconstructPar \n\
        rm -r processor* \n\
        postProcess -func writeCellCentres \n'%(dirname,fname1[i],fname2[j]))
        fp.write('echo dir1-%s-dir2-%s \n\n'%(i,j))
        
fp.close()    
