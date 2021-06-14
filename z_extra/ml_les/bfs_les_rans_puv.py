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
import pickle

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""

# read data from below dir...

lesdir='./les/les_bfs_Re1e4/10'
ransdir='./rans/bfs_rans_Re1e4/500'
#ransdir='./test/bfs_using_meanR/1800'


def get_details(dir_name):
      
    x=[]
    with open(dir_name + '/ccx', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            x.append(line)
    x=np.array(x)        
    x = x.astype(np.float)
           
    y=[]
    with open(dir_name + '/ccy', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            y.append(line)
    y=np.array(y)        
    y = y.astype(np.float)
    
    p=[]
    with open(dir_name + '/pMean', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            p.append(line)
            
    p=np.array(p)        
    p = p.astype(np.float)
            
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
    u=np.array(u)        
    u = u.astype(np.float)
    v=np.array(v)        
    v = v.astype(np.float)               
    w=np.array(w)        
    w = w.astype(np.float) 
                   
            
            
    #filter within xlim,ylim
    I=[]
    for i in range(len(x)):
        if (x[i]<=5.05 and x[i]>=-0.05 and y[i]<=3.05 and y[i]>=-0.05 ):
            I.append(i)
                    
    x=x[I]
    y=y[I]
    u=u[I]
    v=v[I]
    p=p[I]
            
    pD=np.asarray([x,y]).transpose()
         
    xa=np.linspace(1,1,100)
    ya=np.linspace(0,3,100)

    xb=np.linspace(2,2,100)
    yb=np.linspace(0,3,100)

    xc=np.linspace(3,3,100)
    yc=np.linspace(0,3,100)

    xd=np.linspace(4,4,100)
    yd=np.linspace(0,3,100)

    # for u    
    print ('interpolation-1...')      
    f1u=interpolate.LinearNDInterpolator(pD,u)
        
    u1a=np.zeros((len(ya)))
    u1b=np.zeros((len(ya)))
    u1c=np.zeros((len(ya)))
    u1d=np.zeros((len(ya)))
    for j in range(len(ya)):
        u1a[j]=f1u(xa[j],ya[j])
        u1b[j]=f1u(xb[j],yb[j])
        u1c[j]=f1u(xc[j],yc[j])
        u1d[j]=f1u(xd[j],yd[j])

    #for v
    print ('interpolation-2...')      
    f1v=interpolate.LinearNDInterpolator(pD,v)

    v1a=np.zeros((len(ya)))
    v1b=np.zeros((len(ya)))
    v1c=np.zeros((len(ya)))
    v1d=np.zeros((len(ya)))
    for j in range(len(ya)):
        v1a[j]=f1v(xa[j],ya[j])
        v1b[j]=f1v(xb[j],yb[j])
        v1c[j]=f1v(xc[j],yc[j])
        v1d[j]=f1v(xd[j],yd[j])
   
    u_int=[u1a,u1b,u1c,u1d]
    v_int=[v1a,v1b,v1c,v1d]
    y_int=[ya,yb,yc,yd]

    return (xu,xl,pu1,pl1,u_int,v_int,y_int)


xu,xl,pu1,pl1,u1,v1,y1 = get_details(lesdir)
_,_,pu2,pl2,u2,v2,y2   = get_details(ransdir)


plt.figure(2)
plt.plot(u1[0],y1[0],'og',linewidth=2,markevery=1,label='LES')
plt.plot(u2[0],y2[0],'b',linewidth=2,label='RANS')

plt.plot(u1[1]+1,y1[1],'og',linewidth=2,markevery=1)
plt.plot(u2[1]+1,y2[1],'b',linewidth=2)

plt.plot(u1[2]+2,y1[2],'og',linewidth=2,markevery=1)
plt.plot(u2[2]+2,y2[2],'b',linewidth=2)

plt.plot(u1[3]+3,y1[3],'og',linewidth=2,markevery=1)
plt.plot(u2[3]+3,y2[3],'b',linewidth=2)
plt.title('u',fontsize=20)
plt.legend()
#plt.savefig('./p.png',format='png',dpi=100)
plt.show()



plt.figure(3)
plt.plot(v1[0],y1[0],'og',linewidth=2,markevery=1,label='LES')
plt.plot(v2[0],y2[0],'b',linewidth=2,label='RANS')

plt.plot(v1[1]+1,y1[1],'og',linewidth=2,markevery=1)
plt.plot(v2[1]+1,y2[1],'b',linewidth=2)

plt.plot(v1[2]+2,y1[2],'og',linewidth=2,markevery=1)
plt.plot(v2[2]+2,y2[2],'b',linewidth=2)

plt.plot(v1[3]+3,y1[3],'og',linewidth=2,markevery=1)
plt.plot(v2[3]+3,y2[3],'b',linewidth=2)
plt.title('u',fontsize=20)
plt.legend()
#plt.savefig('./p.png',format='png',dpi=100)
plt.show()