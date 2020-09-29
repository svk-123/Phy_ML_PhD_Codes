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

#replace CNN para
path='./data_file/'
data_file='param_216_8p_cnn_tanh_naca4_v1.pkl'
# ['[para_scaled,name,para(unscaled),mm_scaler,info]']
with open(path + data_file, 'rb') as infile:
    result1 = pickle.load(infile)
para=result1[0]
name=result1[1]
name=np.asarray(name)




#del result1
fp=open('removed.dat','w')
path='./data_file/'
for ii in [1]:
    
    data_file='./data_file/naca4_clcd_turb_st_3para.pkl'
    with open( data_file, 'rb') as infile:
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
    new_myinp_cm=[]
    new_myinp_cd=[]
    new_myinp_cl=[]
    new_myinp_reno=[]
    new_myinp_aoa=[]
    new_myinp_para=[]
    new_myname=[]
    
    for j in range(len(myinp_cm)):
        if myname[j] in name:
            ind=np.argwhere(myname[j]==name)
            new_para.append(para[int(ind)])
            
            new_myinp_cm.append(myinp_cm[j])
            new_myinp_cd.append(myinp_cd[j])
            new_myinp_cl.append(myinp_cl[j])
            new_myinp_reno.append(myinp_reno[j])
            new_myinp_aoa.append(myinp_aoa[j])
            new_myinp_para.append(myinp_para[j])            
            new_myname.append(myname[j])   
            
        else:
            print('not in pname %s'%myname[j])
            fp.write('not in pname %s\n'%myname[j])
            
new_para=np.asarray(new_para)
new_myinp_cm=np.asarray(new_myinp_cm)
new_myinp_cd=np.asarray(new_myinp_cd)
new_myinp_cl=np.asarray(new_myinp_cl)
new_myinp_reno=np.asarray(new_myinp_reno)
new_myinp_aoa=np.asarray(new_myinp_aoa)
new_myinp_para=np.asarray(new_myinp_para)
new_myname=np.asarray(new_myname)

mm_scaler=[]

info= '[cm, cd, cl, reno, aoa, para_scaled, name,scaler, info:cl_0p02_cd_0p001]'

data1 = [new_myinp_cm, new_myinp_cd, new_myinp_cl, new_myinp_reno, new_myinp_aoa, new_para, new_myname, mm_scaler, info ]

with open(path+'/naca4_clcd_turb_st_8para_tanh_cnn.pkl', 'wb') as outfile1:
    pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)

fp.close()
