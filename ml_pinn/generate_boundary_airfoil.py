#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:56:16 2019

@author: vino
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas
from os import listdir
from os.path import isfile, join


xy=np.loadtxt('./data_file/naca0006.dat',skiprows=1)

#fp=open('naca0006_bc.dat','w')
#u=0
#v=0
#
#for i in range(len(xy)):
#    
#    fp.write("%f %f %f %f \n"%(X[i],Y[i],U[i],V[i]))
#
#fp.close()