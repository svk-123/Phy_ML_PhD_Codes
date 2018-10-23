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
path='../cnn_airfoil_sf/OF_results/cases_naca_aoa_test'

indir = path

fname = [f for f in listdir(indir) if isdir(join(indir, f))]
fname.sort()
fname=np.asarray(fname)

np.random.seed(1234)
np.random.shuffle(fname)

tmp=[]
for i in range(len(fname)):
    tmp.append(fname[i].split('_')[1])
tmp=np.asarray(tmp)    

foilR=['naca23012','naca66018']

ind_del=[]
for i in range(2):
    if foilR[i] in tmp:
        ind=np.argwhere(tmp==foilR[i])
        ind_del.extend(ind)

fname=np.delete(fname,ind_del,0)
        
coord=[]
for nn in range(len(fname)):
    pts=np.loadtxt('../cnn_airfoil_sf/airfoil_data/foil200_aoa/%s.dat'%fname[nn],skiprows=1)
    coord.append(pts)
 
datafile='./data_file/param_216_16.pkl'
with open(datafile, 'rb') as infile:
    result = pickle.load(infile)
para=result[0][0]    
pname=result[1]
pname=np.asarray(pname)


nname=[]
aoa=[]
for i in range(len(fname)):
    nname.append(fname[i].split('_')[1])
    aoa.append(fname[i].split('_')[3])
nname=np.asarray(nname)  
aoa = np.array(map(float, aoa))


my_para=[]
for i in range(len(nname)):
    if nname[i] in pname:
        ind=np.argwhere(pname==nname[i])
        my_para.append(para[int(ind)])

    else:
        print('not in pname %s'%nname[i])

myinp_x=[]
myinp_y=[]
myinp_para=[]
myinp_aoa=[]

myout_p=[]
myout_u=[]
myout_v=[]

myname=[]

for ii in range(49):
    if ('naca23012' != fname[ii]):
        
        print (ii)
        
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
            if (x[i]<=2.2 and x[i]>=-1.2 and y[i]<=1.2 and y[i]>=-1.2 ):
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
            cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
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
            #plt.savefig('./plotc/%s.eps'%(nname[ii]), format='eps')
            plt.close()
            
        #plot(x,y,u,20,'name')    
        
        myinp_x.append(x)
        myinp_y.append(y)
        myout_p.append(p)
        myout_u.append(u)
        myout_v.append(v)

        paralist=[]
        for k in range(len(x)):
            paralist.append(my_para[ii])
        paralist=np.asarray(paralist)
    
        aoalist=[]
        for k in range(len(x)):
            aoalist.append(aoa[ii])
        aoalist=np.asarray(aoalist)

        namelist=[]
        for k in range(1):
            namelist.append(nname[ii])
        namelist=np.asarray(namelist)

        myinp_para.append(paralist)
        myinp_aoa.append(aoalist)
        myname.append(namelist)
        
#save file
filepath='./data_file'
      
# ref:[x,y,z,ux,uy,uz,k,ep,nut]
info=['myinp_x, myinp_y, myinp_para, myinp_aoa, myout_p, myout_u, myout_v, coord, myname, fname, info']

data1 = [myinp_x, myinp_y, myinp_para, myinp_aoa, myout_p, myout_u, myout_v, coord, myname, fname, info ]

with open(filepath+'/foil_aoa_nn_test_ts_p16_NT.pkl', 'wb') as outfile1:
    pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)

    
        
