#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
reads and plots the compasion of LES and RANS airfoils


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

# read data from below dir...

lesdir='./les/les_naca_0012_aoa_6/10'
ransdir='./rans/rans_naca_0012_aoa_6/330/'

def get_details(dir_name):
      
    x=[]
    with open(dir_name + '/ccx', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            x.append(line)
    x = np.array(map(float, x))
           
    y=[]
    with open(dir_name + '/ccy', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            y.append(line)
    y = np.array(map(float, y))
    
    p=[]
    with open(dir_name + '/pMean', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            p.append(line)
    p = np.array(map(float, p))
            
    # load velocity
    u=[]
    v=[]
    w=[]
    with open(dir_name + '/UMean', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            line=line.replace("(","")
            line=line.replace(")","")        
            a, b, c = (item.strip() for item in line.split(' ', 3))
            u.append(a), v.append(b), w.append(c)
    u = np.array(map(float, u))
    v = np.array(map(float, v))
    w = np.array(map(float, w))
                   
            
            
    #filter within xlim,ylim
    I=[]
    for i in range(len(x)):
        if (x[i]<=2.2 and x[i]>=-0.6 and y[i]<=0.6 and y[i]>=-0.6 ):
            I.append(i)
                    
    x=x[I]
    y=y[I]
    u=u[I]
    v=v[I]
    p=p[I]
            
    pD=np.asarray([x,y]).transpose()
    
    
    co=np.loadtxt('./data_file/coord/n0012.dat', skiprows=1)
    co=co[:,0:2]  
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx  
    a0=find_nearest(co[:,0],0)
        
    xu=co[:a0+1,0]
    yu=co[:a0+1,1]
    if(yu[0] <=0.001):
        yu[0]=0.001
            
    xl=co[a0:,0]
    yl=co[a0:,1]
    if(yl[-1:] >=-0.001):
        yl[-1:]=-0.001  
    
    
    #for -p
    print 'interpolation-1...'      
    f1p=interpolate.LinearNDInterpolator(pD,p)
            
    pu1=np.zeros(len(xu))
    for j in range(len(xu)):
        pu1[j]=f1p(xu[j],yu[j])
    pl1=np.zeros(len(xl))
    for j in range(len(xl)):
        pl1[j]=f1p(xl[j],yl[j])

    return (xu,xl,pu1,pl1)

xu,xl,pu1,pl1=get_details(lesdir)
_,_,pu2,pl2=get_details(ransdir)


plt.figure(1)
plt.plot(xu,pu1,'og',linewidth=2,markevery=2,label='LES')
plt.plot(xl,pl1,'ob',linewidth=2,markevery=2) 
plt.plot(xu,pu2,'g',linewidth=2,label='RANS')
plt.plot(xl,pl2,'b',linewidth=2) 
plt.title('Pressure',fontsize=20)
plt.legend()
#plt.savefig('./p.png',format='png',dpi=100)
plt.show()
   
