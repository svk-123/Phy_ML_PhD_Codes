#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 02:28:22 2019

@author: vino
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import pickle




fp=open('./data_file/cy_inout_20.dat','w')

N=20

x=np.linspace(-5,-5,N)
y=np.linspace(-5,5,N)

for i in range(len(x)):
    fp.write('%f %f 1.0 1e-12 \n'%(x[i],y[i]))


#x=np.linspace(5,5,N)
#y=np.linspace(-3,3,N)
#
#for i in range(len(x)):
#    fp.write('%f %f 1.0 1e-12 \n'%(x[i],y[i]))

x=np.linspace(-5,5,N)
y=np.linspace(-5,-5,N)

for i in range(len(x)):
    fp.write('%f %f 1.0 1e-12 \n'%(x[i],y[i]))


x=np.linspace(-5,5,N)
y=np.linspace(5,5,N)

for i in range(len(x)):
    fp.write('%f %f 1.0 1e-12 \n'%(x[i],y[i]))


        
fp.close()    
    
