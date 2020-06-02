#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:49:29 2017

This code process OF data and exports as .pkl to prepData file
for TBNN. prepData reads .pkl and process further

@author: vino
"""
# imports
import os
import glob

import numpy as np
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate
from os import listdir
from os.path import isfile,isdir, join

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""
# duct - list
flist=['Re100','Re200','Re300','Re400','Re500','Re600','Re700','Re800','Re900','Re1000']
#flist=['Re100']
Re=[100,200,300,400,500,600,700,800,900,1000]

# read data from below dir...
path='./cases_v1/'

for ii in range(len(flist)):
    
    casedir= path +'%s'%(flist[ii])
                
    #need to find max time later....
    yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
    yname = np.asarray(yname)
    yname.sort()
    yname=yname[:-3].astype(np.int) 
    ymax=int(yname.max())
    
    x=[]
    with open(path+'%s/%s/ccx'%(flist[ii],ymax), 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            x.append(line)
    x = np.array(map(float, x))
   
    y=[]
    with open(path+'%s/%s/ccy'%(flist[ii],ymax), 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            y.append(line)
    y = np.array(map(float, y))
    
    z=[]
    with open(path+'%s/%s/ccz'%(flist[ii],ymax), 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            z.append(line)
    z = np.array(map(float, z))
    
    p=[]
    with open(path+'%s/%s/p'%(flist[ii],ymax), 'r') as infile:
        data0=infile.readlines()
        print (data0[20])
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            p.append(line)
    p = np.array(map(float, p))
    
    
    # load velocity
    u=[]
    v=[]
    w=[]
    with open(path+'%s/%s/U'%(flist[ii],ymax), 'r') as infile:
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
    
    relist=[]
    for k in range(len(u)):
        relist.append(Re[ii])
    relist=np.asarray(relist)    
    
    
    
    '''tmp1=[]
    for iii in range(len(x)):
    
        if (x[iii] >=0.45 and x[iii] <=0.55):
            tmp1.append(iii)
            
    tmp=tmp1'''
    
    tmp2=[]
    for iii in range(len(y)):
    
        if (y[iii] >=0.45 and y[iii] <=0.55):
            tmp2.append(iii) 
    
    tmp=tmp2
    

    
    
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
          
    plot(x,y,u,20,'name')    
    
    
    #save file
    import cPickle as pickle
    filepath='./data/'
    suff='r0_l7'
    # ref:[x,y,z,ux,uy,uz,k,ep,nut]
    
    
    name=['0-x','1-y','2-re','3-u','4-v','5-p']
    data1 = [x[tmp],y[tmp],relist[tmp],u[tmp],v[tmp],p[tmp],name]
    with open(filepath+'cavity_%s_part_y.pkl'%flist[ii], 'wb') as outfile1:
        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)

    
        
