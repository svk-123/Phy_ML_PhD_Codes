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


xl=np.linspace(0,0,100)
xr=np.linspace(1,1,100)
xt=np.linspace(0,1,100)
xb=np.linspace(0,1,100)

yl=np.linspace(0,1,100)
yr=np.linspace(0,1,100)
yt=np.linspace(1,1,100)
yb=np.linspace(0,0,100)

ul=np.linspace(0,0,100)
ur=np.linspace(0,0,100)
ut=np.linspace(1,1,100)
ub=np.linspace(0,0,100)

vl=np.linspace(0,0,100)
vr=np.linspace(0,0,100)
vt=np.linspace(0,0,100)
vb=np.linspace(0,0,100)

X=np.concatenate((xl,xr,xt,xb),axis=0)
Y=np.concatenate((yl,yr,yt,yb),axis=0)
U=np.concatenate((ul,ur,ut,ub),axis=0)
V=np.concatenate((vl,vr,vt,vb),axis=0)

plt.figure()
plt.plot(X,Y,'o')
plt.show()

fp=open('ldc_bc.dat','w')

for i in range(len(X)):
    fp.write("%f %f %f %f \n"%(X[i],Y[i],U[i],V[i]))

fp.close()