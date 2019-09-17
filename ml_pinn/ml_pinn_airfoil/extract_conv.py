#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:17:29 2019

@author: vino
"""
import time
start_time = time.time()

# Python 3.5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas
from os import listdir
from os.path import isfile, join

tmp0=[]
it=[]
l=[]
l1=[]
l2=[]

path='./tf_model_sf_re_aoa/case_2_nn_8x500_100pts_relu/'
with open( path + 'nn-100pts.o8949673', 'r') as infile:
    data0=infile.readlines()
    for line in data0:
        if 'It:' in line:
            tmp0.append(line)
            
    for i in range(len(tmp0)):
        tmp1=tmp0[i].split(',')
        it.append(np.float(tmp1[0].split(':')[1]))
        l.append(np.float(tmp1[1].split(':')[1]))        
        l1.append(np.float(tmp1[2].split(':')[1]))                
        l2.append(np.float(tmp1[3].split(':')[1]))        


fp=open(path + 'conv_nn.dat','w+')

fp.write('It, Loss, Loss-1 (MSE), Loss-2 (Res) \n')
for i in range(len(it)):
    fp.write('%f, %0.6f, %0.6f, %0.6f \n' %(it[i],l[i],l1[i],l2[i]))
    
fp.close()

     