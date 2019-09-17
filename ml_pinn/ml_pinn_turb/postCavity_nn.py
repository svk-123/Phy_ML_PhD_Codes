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


"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""
## duct - list
#flist=['Re100','Re200','Re300','Re400','Re500','Re600','Re700','Re800','Re900','Re1000','Re1200','Re1500',\
#       'Re2000','Re3000','Re4000','Re5000','Re6000','Re7000','Re8000','Re9000','Re10000']
##flist=['Re100']
#Re=[100,200,300,400,500,600,700,800,900,1000,1200,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000]


flist=['Re20000']
Re=[2000]

# read data from below dir...
path='./ldc/'

for ii in range(len(flist)):

        casedir= path +'/%s/%s'%(foil[ii],tmp[ii])
   
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname[:-3].astype(np.int) 
        ymax=int(yname.max())
        

        fp1.write('%s-%s\n'%(ii,casedir))  
	fp1.write('	yname:%s\n'%(yname)) 
	fp1.write('	ymax:%s\n'%(ymax))     



    x=[]
    with open(path+'%s/748/ccx'%flist[ii], 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            x.append(line)
    x = np.array(map(float, x))
   
    y=[]
    with open(path+'%s/748/ccy'%flist[ii], 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            y.append(line)
    y = np.array(map(float, y))
    
    z=[]
    with open(path+'%s/748/ccz'%flist[ii], 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            z.append(line)
    z = np.array(map(float, z))
    
    p=[]
    with open(path+'%s/748/p'%flist[ii], 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            p.append(line)
    p = np.array(map(float, p))

    nutilda=[]
    with open(path+'%s/748/nuTilda'%flist[ii], 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            nutilda.append(line)
    nutilda = np.array(map(float, nutilda))    

    nut=[]
    with open(path+'%s/748/nut'%flist[ii], 'r') as infile:
        data0=infile.readlines()
        npt=int(data0[20])
        for line in data0[22:22+npt]:
            nut.append(line)
    nut = np.array(map(float, nut))  
    
    # load velocity
    u=[]
    v=[]
    w=[]
    with open(path+'%s/748/U'%flist[ii], 'r') as infile:
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
    filepath=path
    suff='r0_l7'
    # ref:[x,y,z,ux,uy,uz,k,ep,nut]
    name=['x,y,relist,u,v,p,nut,nutilda,name']
    data1 = [x,y,relist,u,v,p,nut,nutilda,name]
    with open(filepath+'cavity_%s.pkl'%flist[ii], 'wb') as outfile1:
        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)

    
        
