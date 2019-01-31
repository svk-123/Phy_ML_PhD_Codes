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
data_file='param_216_5c_tanh_16.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)
para=result1[0]
name=result1[1]
name=np.asarray(name)

del result1

path='./data_file/'
for ii in [1]:
    
    data_file='foil_nacan_lam_tr_%s.pkl'%ii
    with open(path + data_file, 'rb') as infile:
        result = pickle.load(infile)
    
    myinp_x=result[0]
    myinp_y=result[1]
    myinp_para=result[2]
    myinp_re=result[3]
    myinp_aoa=result[4]
    myout_p=result[5]
    myout_u=result[6]
    myout_v=result[7]
    coord=result[8]
    myname=result[9]
    info=result[10]
    myname=np.asarray(myname)
    
    del result

    new_para=[]
    for j in range(len(myinp_x)):
        if myname[j] in name:
            ind=np.argwhere(myname[j]==name)
            new_para.append(para[int(ind)])

        else:
            print('not in pname %s'%myname[j])


#    # ref:[x,y,z,ux,uy,uz,k,ep,nut]
#    info=['myinp_x, myinp_y, new_para, myinp_re, myinp_aoa, myout_p, myout_u, myout_v, coord, mynae, info: new tanh para is used']
#    data1 = [myinp_x, myinp_y, new_para, myinp_re, myinp_aoa, myout_p, myout_u, myout_v, coord, myname, info ]
#    with open(path+'/foil_nacan_np_lam_tr_%s.pkl'%(ii), 'wb') as outfile1:
#        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)

     
