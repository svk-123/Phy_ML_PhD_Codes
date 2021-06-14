#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:49:29 2017

This code reads openfoam LES data and 
interpolates LES variable for x,y,x of RANS in pkl. 

@author: vino
"""
# imports
import os
import glob

import numpy as np
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate


"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""
# duct - list
# read data from below dir...
path='./les/les_naca_0012_aoa_6/10'
pathR='./rans/rans_naca_0012_aoa_6/330'

for ii in range(1):
    
    x=[]
    with open(path+'/ccx', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            x.append(line)
    x = np.array(map(float, x))
   
    y=[]
    with open(path+'/ccy', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            y.append(line)
    y = np.array(map(float, y))
    
    z=[]
    with open(path+'/ccz', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            z.append(line)
    z = np.array(map(float, z))


    xR=[]
    with open(pathR+'/ccx', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            xR.append(line)
    xR = np.array(map(float, xR))
   
    yR=[]
    with open(pathR+'/ccy', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            yR.append(line)
    yR = np.array(map(float, yR))



    nut=[]
    with open(path+'/nut', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            nut.append(line)
    nut = np.array(map(float, nut))
    
    
    #load reynols stress
    rxx=[]
    rxy=[]
    rxz=[]
    ryy=[]
    ryz=[]
    rzz=[]
    
    with open(path+'/turbulenceProperties:R', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            line=line.replace("(","")
            line=line.replace(")","")        
            a, b, c,d,e,f = (item.strip() for item in line.split(' ', 6))
            rxx.append(a), rxy.append(b), rxz.append(c),ryy.append(d), ryz.append(e), rzz.append(f)
    rxx = np.array(map(float, rxx))
    rxy = np.array(map(float, rxy))
    rxz = np.array(map(float, rxz))
    ryy = np.array(map(float, ryy))
    ryz = np.array(map(float, ryz))
    rzz = np.array(map(float, rzz))
    

    #interpolatiom
    print ('interpolating')
    pD=np.asarray([x, y]).transpose()

    fuuD=interpolate.LinearNDInterpolator(pD, rxx)
    fuvD=interpolate.LinearNDInterpolator(pD, rxy)
    fuwD=interpolate.LinearNDInterpolator(pD, rxz)
    fvvD=interpolate.LinearNDInterpolator(pD, ryy)
    fvwD=interpolate.LinearNDInterpolator(pD, ryz)
    fwwD=interpolate.LinearNDInterpolator(pD, rzz)

    rxxi = np.zeros((len(xR)))
    rxyi = np.zeros((len(xR)))
    rxzi = np.zeros((len(xR)))
    ryyi = np.zeros((len(xR)))
    ryzi = np.zeros((len(xR)))
    rzzi = np.zeros((len(xR)))


    for i in range(len(xR)):
        rxxi[i]=fuuD(xR[i],yR[i])
        rxyi[i]=fuvD(xR[i],yR[i])
        rxzi[i]=fuwD(xR[i],yR[i])
        ryyi[i]=fvvD(xR[i],yR[i])
        ryzi[i]=fvwD(xR[i],yR[i])
        rzzi[i]=fwwD(xR[i],yR[i])


            
    #plot
    def plot(xp,yp,zp,nc,name):
        pyplot.figure(figsize=(6, 5), dpi=100)
        #cp = pyplot.tricontour(ys, zs, pp,nc)
        cp = pyplot.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
        #cp = pyplot.tripcolor(ys, zs, pp)
        #cp = pyplot.scatter(ys, zs, pp)
        #pyplot.clabel(cp, inline=False,fontsize=8)
        pyplot.colorbar()
        pyplot.title(name)
        pyplot.xlabel('Z ')
        pyplot.ylabel('Y ')
        #pyplot.savefig(name, format='png', dpi=100)
        pyplot.show()
    
       
    #plot(z,y,ux,20,'name')    
    
    #save file
    import cPickle as pickle
    filepath='./data_file/les_data'

    info1=['rxxi,rxyi,rxzi,ryyi,ryzi,rzzi,info1']
    data1 = [rxxi,rxyi,rxzi,ryyi,ryzi,rzzi,info1]
    with open(filepath+'/les_data1.pkl', 'wb') as outfile1:
        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)
    

        
        
