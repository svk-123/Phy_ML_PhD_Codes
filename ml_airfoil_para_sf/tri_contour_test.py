#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 19:49:18 2019

@author: vino
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm
import pandas
from os import listdir
from os.path import isfile, join
import cPickle as pickle

with open('tri_tmp.pkl', 'rb') as infile:
    result = pickle.load(infile)

xp=result[0]  
yp=result[1] 
zp=result[2] 
xx=result[3]
yy=result[4]


plt.figure(figsize=(8, 4), dpi=100)
   
zz=np.zeros(len(xx))
zz[:]=np.nan
#zz[:]=np.ma.masked
    
xp=np.concatenate((xp,xx),axis=0)    
yp=np.concatenate((yp,yy),axis=0) 
zp_tmp=np.concatenate((zp,zz),axis=0)
mask = np.isnan(zp_tmp)

zz=np.zeros(len(xx))
zp=np.concatenate((zp,zz),axis=0)
   
triang = tri.Triangulation(xp, yp)
#mask = np.isnan(zp)
#triang.set_mask(mask)

plt.tricontourf(triang,zp,20,cmap='gray')    
#cp = plt.tricontour(xp,yp,zp,20,cmap=cm.jet,mask=mask)
    
plt.xlim(-0.5,2)
plt.ylim(-0.5,0.5)
plt.xlabel('X ',fontsize=20)
plt.ylabel('Y ',fontsize=20)
plt.subplots_adjust(top = 0.95, bottom = 0.15, right = 0.98, left = 0.14, hspace = 0, wspace = 0)
plt.savefig('tmp.png',format='png',dpi=200)
plt.show()
plt.close()