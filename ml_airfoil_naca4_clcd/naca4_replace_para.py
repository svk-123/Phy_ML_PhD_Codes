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
from matplotlib import cm
import pandas as pd
from scipy import interpolate
from os import listdir
from os.path import isfile,isdir, join
import cPickle as pickle

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""

# ref:[data,name]
path='./data_file/'
data_file='param_naca4_tanh_8_v1.pkl'
# ['[para_scaled,name,para(unscaled),mm_scaler,info]']
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)
para=result1[0]
name=result1[1]
mm_scaler=result1[3]
name=np.asarray(name)

#del result1

path='./data_file/'
for ii in [1]:
    
    data_file='naca4_clcd_turb_st_3para.pkl'
    with open(path + data_file, 'rb') as infile:
        result = pickle.load(infile)
    
    myinp_cm=result[0]
    myinp_cd=result[1]
    myinp_cl=result[2]
    myinp_reno=result[3]
    myinp_aoa=result[4]
    myinp_para=result[5]
    myname=result[6]

    myname=np.asarray(myname)
    
    #del result

    new_para=[]
    for j in range(len(myinp_cm)):
        if myname[j] in name:
            ind=np.argwhere(myname[j]==name)
            new_para.append(para[int(ind)])

        else:
            print('not in pname %s'%myname[j])
            
new_para=np.asarray(new_para)

info= '[cm, cd, cl, reno, aoa, para_scaled, name,scaler, info]'

data1 = [myinp_cm, myinp_cd, myinp_cl, myinp_reno, myinp_aoa, new_para, myname, mm_scaler, info ]

with open(path+'/naca4_clcd_turb_st_8para.pkl', 'wb') as outfile1:
    pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)


