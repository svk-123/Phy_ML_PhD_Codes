#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:31:27 2017

@author: vino
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 19:35:45 2017

@author: vino
"""

# imports
import os
import glob

import numpy as np
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate
import cPickle as pickle

# time
import time
start_time = time.time()
                      


#flist=['Re3500']
# import functions
from calc_rans import calc_rans
from calc_dns import calc_dns
from plot_tau import plot,plotD

dataframe = pd.read_csv('./channel/channel.txt', sep='\s+', header=None, skiprows=1)
dataset = dataframe.values
data=np.asarray(dataset)

#info
#tke, epsilon, \
#grad_u_00, grad_u_01, grad_u_02, grad_u_10, grad_u_11, grad_u_12, grad_u_20, grad_u_21, grad_u_22,
#uu_00, uu_01, uu_02, uu_10, uu_11, uu_12, uu_20, uu_21, uu_22

k = data[:, 0]
ep = data[:, 1]
grad_u_flat = data[:, 2:11]
stresses_flat = data[:, 11:]

UUDi,aD,bD = calc_dns(stresses_flat)

L,T1m,T2m,T3m,T4m,T5m,T6m,S,R = calc_rans(k,ep,grad_u_flat)