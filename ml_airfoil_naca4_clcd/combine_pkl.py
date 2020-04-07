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
from os.path import isfile,isdir, join
import cPickle as pickle

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""

# ref:[data,name]
cm=[]
cd=[]
cl=[]
reno=[]
aoa=[]
para=[]
foil=[]

path='./data_file_to_combine/clcd_pkls_cl_0p02_cd_0p001/'
for ii in range(49):
    
    data_file='gen_clcd_turb_st_%s.pkl'%ii
    with open(path + data_file, 'rb') as infile:
        result = pickle.load(infile)
    
        cm.extend(result[0])
        cd.extend(result[1])
        cl.extend(result[2])
        reno.extend(result[3])
        aoa.extend(result[4])
        para.extend(result[5])
        foil.extend(result[6])

cm=np.asarray(cm)
cd=np.asarray(cd)
cl=np.asarray(cl)
reno=np.asarray(reno) 
aoa=np.asarray(aoa) 
para=np.asarray(para) 
foil=np.asarray(foil) 
     
# ref:[x,y,z,ux,uy,uz,k,ep,nut]
info= '[cm, cd, cl, reno, aoa, para, name, info-converged ]'

data1 = [cm, cd, cl, reno, aoa, para, foil, info ]

with open(path+'/gen_clcd_turb_st_all_0para.pkl', 'wb') as outfile1:
    pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)