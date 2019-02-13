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
data_file='param_216_tanh_16_v1.pkl'
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)
para=result1[0]
name=result1[1]
name=np.asarray(name)

del result1

path='./data_file/'
for ii in [1,2]:
    
    data_file='foil_aoa_nn_nacan_lam_ts_%s.pkl'%ii
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
    for k in range(len(myinp_x)):
        tmp_para=[]
        for j in range(len(myinp_x[k])):
            if myname[k][j] in name:
                ind=np.argwhere(myname[k][j]==name)
                tmp_para.append(para[int(ind)])

            else:
                print('not in pname %s'%myname[j])
                
        new_para.append(np.asarray(tmp_para))

    # ref:[x,y,z,ux,uy,uz,k,ep,nut]
    info=['myinp_x, myinp_y, new_para, myinp_re, myinp_aoa, myout_p, myout_u, myout_v, coord, mynae, info: new tanh para is used']
    data1 = [myinp_x, myinp_y, new_para, myinp_re, myinp_aoa, myout_p, myout_u, myout_v, coord, myname, info ]
    with open(path+'/foil_aoa_nn_nacan_lam_np_ts_%s.pkl'%(ii), 'wb') as outfile1:
        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)

     
