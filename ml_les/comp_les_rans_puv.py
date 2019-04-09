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

lesdir='./les/les_s805_Re_1e6_aoa_14/9.98'
ransdir='./rans/rans_s805_Re1e6_aoa_14/475.5'

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
    
    
    co=np.loadtxt('./data_file/coord/s805.dat', skiprows=1)
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
    print ('interpolation-1...')      
    f1p=interpolate.LinearNDInterpolator(pD,p)
            
    pu1=np.zeros(len(xu))
    for j in range(len(xu)):
        pu1[j]=f1p(xu[j],yu[j])
    pl1=np.zeros(len(xl))
    for j in range(len(xl)):
        pl1[j]=f1p(xl[j],yl[j])


    #plot u,v
    dl=int(len(co[:,0])/2)
    
    a0=find_nearest(co[:dl+2,0],0)
    a5=find_nearest(co[:dl+2,0],0.5)
    
    xa=np.linspace(co[a0,0],co[a0,0],50)
    ya=np.linspace(co[a0,1],0.5,50)

    xb=np.linspace(co[a5,0],co[a5,0],50)
    yb=np.linspace(co[a5,1],0.5,50)

    xc=np.linspace(co[0,0],co[0,0],50)
    yc=np.linspace(co[0,1],0.5,50)

    xd=np.linspace(1.5,1.5,50)
    yd=np.linspace(co[0,1],0.99,50)


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


plt.figure(1)
plt.plot(xu,pu1,'og',linewidth=2,markevery=2,label='LES')
plt.plot(xl,pl1,'ob',linewidth=2,markevery=2) 
plt.plot(xu,pu2,'g',linewidth=2,label='RANS')
plt.plot(xl,pl2,'b',linewidth=2) 
plt.title('Pressure',fontsize=20)
plt.legend()
#plt.savefig('./p.png',format='png',dpi=100)
plt.show()

 
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