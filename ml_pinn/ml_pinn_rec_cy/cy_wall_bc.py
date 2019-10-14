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


theta=np.linspace(0,360,20)*np.pi/180.

x=0.5*np.cos(theta)
y=0.5*np.sin(theta)

fp=open('cy_wall_bc_20.dat','w')

for i in range(len(x)):
    fp.write('%f %f 1e-12 1e-12 \n'%(x[i],y[i]))
    
    
fp.close()    
    
