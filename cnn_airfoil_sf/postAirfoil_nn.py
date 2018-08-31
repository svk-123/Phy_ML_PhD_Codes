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
path='./OF_results/cases_naca_aoa'

indir = path

fname = [f for f in listdir(indir) if isdir(join(indir, f))]
fname.sort()

#nname=[]
#for i in range(len(fname)):
#    nname.append(fname[i].split('_')[1])

nname=fname

#bor(yx),ins(xy)
datafile='./airfoil_data/foil_aoa_df.pkl'
with open(datafile, 'rb') as infile:
    result = pickle.load(infile)
    
print result[-1:]


inp=result[2]
bor=result[3]
ins=result[4]
name=result[5]

coord=[]
for nn in range(len(fname)):
    pts=np.loadtxt('./airfoil_data/foil200_aoa/%s.dat'%fname[nn],skiprows=1)
    coord.append(pts)


myinp=[]
myout_p=[]
myout_u=[]
myout_v=[]
myco=[]
mybor=[]
myins=[]

for ii in range(len(fname)):
    if 'naca23012' not in fname[ii]:
        
        print ii
        
        casedir= path +'/%s'%fname[ii]
                
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
                
        
        x=[]
        with open(casedir +'/500/ccx', 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                x.append(line)
        x = np.array(map(float, x))
       
        y=[]
        with open(casedir +'/500/ccy', 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                y.append(line)
        y = np.array(map(float, y))
        
        z=[]
        with open(casedir +'/500/ccz', 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                z.append(line)
        z = np.array(map(float, z))
        
        p=[]
        with open(casedir +'/500/p', 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                p.append(line)
        p = np.array(map(float, p))
        
        
        # load velocity
        u=[]
        v=[]
        w=[]
        with open(casedir +'/500/U', 'r') as infile:
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
            if (x[i]<=3.2 and x[i]>=-2.2 and y[i]<=2.2 and y[i]>=-2.2 ):
                I.append(i)
                
                
        x=x[I]
        y=y[I]
        z=z[I]
        u=u[I]
        v=v[I]
        w=w[I]
        p=p[I]
            
        
        #plot
        def plot(xp,yp,zp,nc,name):
            plt.figure(figsize=(3, 4))
            #cp = pyplot.tricontour(ys, zs, pp,nc)
            cp = plt.contourf(xp,yp,zp,nc,cmap=cm.jet)
            #cp=pyplot.tricontourf(x1,y1,z1)
            #cp=pyplot.tricontourf(x2,y2,z2)   
            
            #cp = pyplot.tripcolor(xp, yp, zp)
            #cp = pyplot.scatter(ys, zs, pp)
            #pyplot.clabel(cp, inline=False,fontsize=8)
            plt.xlim(-1,2)
            plt.ylim(-1,1)    
            plt.axis('off')
            #plt.grid(True)
            #patch.set_facecolor('black')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.savefig('./plotc/%s.eps'%(nname[ii]), format='eps')
            plt.close()
            
        #plot(x,y,u,20,'name')    
        
        
        xy=np.concatenate((x[:,None],y[:,None]),axis=1)
        
        # y axis inverted for correct imshow
        grid_x, grid_y = np.meshgrid(np.linspace(-1,2,216), np.linspace(1,-1,288))
        
        ui = interpolate.griddata(xy, u, (grid_x, grid_y), method='linear')
        vi = interpolate.griddata(xy, v, (grid_x, grid_y), method='linear')
        pi = interpolate.griddata(xy, p, (grid_x, grid_y), method='linear')
        
        if (np.isnan([ui,vi,pi]).any() == True):
            raise ValueError('Error')            
        
        #z1n=z1.copy()
        #plt.imshow(ui)
        plot(grid_x, grid_y,ui,20,'name')  
        
        ind=name.index('%s'%nname[ii])
        
        if(name[ind] != nname[ii]):
            raise ValueError('Error')
        
        myout_p.append(pi)
        myout_u.append(ui)
        myout_v.append(vi)
        
        myco.append(coord[ind])        
        myinp.append(inp[ind])
        mybor.append(bor[ind])
        myins.append(ins[ind])
        
        
#save file
filepath='./airfoil_data'
      
# ref:[x,y,z,ux,uy,uz,k,ep,nut]
info=['0-myinp, 1-myout_p, 2-myout_u, 3-myout_v, 4-myco, 5-mybor, 6-myins, 7-nname']

data1 = [myinp, myout_p, myout_u, myout_v, myco, mybor, myins, nname, info ]

with open(filepath+'/foil_aoa_inoutxxx.pkl', 'wb') as outfile1:
    pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)

    
        
