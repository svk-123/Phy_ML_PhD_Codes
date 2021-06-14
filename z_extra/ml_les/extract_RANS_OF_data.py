#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:49:29 2017

This code reads openfoam RANS data and extracts the variable in pkl. 

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
path='./rans/rans_naca_0012_aoa_6/330'

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


    p=[]
    with open(path+'/p', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            p.append(line)
    p = np.array(map(float, p))

    
    # load velocity
    u=[]
    v=[]
    w=[]
    with open(path+'/U', 'r') as infile:
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
    

    #load pressure grad
    px=[]
    py=[]
    pz=[]
    with open(path+'/grad(p)', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            line=line.replace("(","")
            line=line.replace(")","")        
            a, b, c = (item.strip() for item in line.split(' ', 3))
            px.append(a), py.append(b), pz.append(c)
    px = np.array(map(float, px))
    py = np.array(map(float, py))
    pz = np.array(map(float, pz))
   
    nutilda=[]
    with open(path+'/nuTilda', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            nutilda.append(line)
    nutilda = np.array(map(float, nutilda))
    
    nut=[]
    with open(path+'/nut', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            nut.append(line)
    nut = np.array(map(float, nut))
    
    #load vel.grad
    ux=[]
    uy=[]
    uz=[]
    vx=[]
    vy=[]
    vz=[]
    wx=[]
    wy=[]
    wz=[]
    with open(path+'/grad(U)', 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            line=line.replace("(","")
            line=line.replace(")","")        
            a, b, c,d,e,f,g,h,i = (item.strip() for item in line.split(' ', 9))
            ux.append(a), uy.append(b), uz.append(c),vx.append(d), vy.append(e), vz.append(f),\
                     wx.append(g), wy.append(h), wz.append(i)
    ux = np.array(map(float, ux))
    uy = np.array(map(float, uy))
    uz = np.array(map(float, uz))
    vx = np.array(map(float, vx))
    vy = np.array(map(float, vy))
    vz = np.array(map(float, vz))
    wx = np.array(map(float, wx))
    wy = np.array(map(float, wy))
    wz = np.array(map(float, wz))
    
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
    filepath='./data_file/rans_data'

    info1=['x,y,z,u,v,w,nutilda,nut']
    data1 = [x,y,z,u,v,w,nutilda,nut,info1]
    with open(filepath+'/rans_data1.pkl', 'wb') as outfile1:
        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)
    
    info2=['px,py,pz,ux,uy,uz,vx,vy,vz,wx,wy,wz']    
    data2 = [px,py,pz,ux,uy,uz,vx,vy,vz,wx,wy,wz,info2]
    with open(filepath+'/rans_data2.pkl', 'wb') as outfile2:
        pickle.dump(data2, outfile2, pickle.HIGHEST_PROTOCOL)
    
    info3=['rxx,rxy,rxz,ryy,ryz,rzz'] 
    data3 = [rxx,rxy,rxz,ryy,ryz,rzz,info3]
    with open(filepath+'/rans_data3.pkl', 'wb') as outfile3:
        pickle.dump(data3, outfile3, pickle.HIGHEST_PROTOCOL)
        
        
