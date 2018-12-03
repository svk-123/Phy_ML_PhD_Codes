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
path='./foam_case'

indir = path

fname_1 = [f for f in listdir(indir) if isdir(join(indir, f))]
fname_1.sort()
fname_1=np.asarray(fname_1)

#np.random.seed(1234)
#np.random.shuffle(fname)

fname_2=[]
for i in range(len(fname_1)):
    dir2=indir + '/%s'%fname_1[i]
    tmp=[f for f in listdir(dir2) if isdir(join(dir2, f))]
    fname_2.append(tmp)
    


tmp=[]
foil=[]
for i in range(len(fname_1)):
    for j in range(len(fname_2[i])):
        tmp.append(fname_2[i][j])
        foil.append(fname_2[i][j].split('_')[0])
tmp=np.asarray(tmp)    
foil=np.asarray(foil)

#foilR=['naca23012xx','naca66018xx']
#ind_del=[]
#for i in range(2):
#    if foilR[i] in tmp:
#        ind=np.argwhere(tmp==foilR[i])
#        ind_del.extend(ind)
#fname=np.delete(fname,ind_del,0)
       
coord=[]
for nn in range(len(foil)):
    pts=np.loadtxt('../cnn_airfoil_sf/airfoil_data/coord_seligFmt_formatted/%s.dat'%foil[nn],skiprows=1)
    coord.append(pts)
 
datafile='./data_file/param_216_16.pkl'
with open(datafile, 'rb') as infile:
    result = pickle.load(infile)
para=result[0][0]    
pname=result[1]
pname=np.asarray(pname)

aoa=[]
reno=[]
for i in range(len(tmp)):
    reno.append(tmp[i].split('_')[1])    
    aoa.append(tmp[i].split('_')[2])

reno=np.array(map(float, reno))
aoa = np.array(map(float, aoa))

my_para=[]
for i in range(len(foil)):
    if foil[i] in pname:
        ind=np.argwhere(pname==foil[i])
        my_para.append(para[int(ind)])

    else:
        print('not in pname %s'%foil[i])

st  = [0,200,1000]
end = [200,len(foil)]
fp=open('foil_error.dat','w+')

for jj in range(1):


    myinp_x=[]
    myinp_y=[]
    myinp_para=[]
    myinp_aoa=[]
    myinp_re=[]

    myout_p=[]
    myout_u=[]
    myout_v=[]

    myname=[]
    myfname=[]
    foamxy=[]
    
    for ii in range(5):
        print ii
        
        casedir= path +'/%s/%s'%(foil[ii],tmp[ii])
                
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname[:-3].astype(np.int) 
        ymax=int(yname.max())
        
        x=[]
        with open(casedir +'/%s/Cx'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                x.append(line)
        x = np.array(map(float, x))
       
        y=[]
        with open(casedir +'/%s/Cy'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                y.append(line)
        y = np.array(map(float, y))
        
        z=[]
        with open(casedir +'/%s/Cz'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                z.append(line)
        z = np.array(map(float, z))
        

        
        # load velocity
        u=[]
        v=[]
        w=[]
        with open(casedir +'/%s/U'%ymax, 'r') as infile:
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
              
        
        #load vel.grad
        ux=[]
        uy=[]
        uz=[]
        vx=[]
        vy=[]
        vz=[]

        with open(casedir +'/%s/grad(U)'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c,d,e,f,g,h,i = (item.strip() for item in line.split(' ', 9))
                ux.append(a), uy.append(b), uz.append(c),vx.append(d), vy.append(e), vz.append(f)
        ux = np.array(map(float, ux))
        uy = np.array(map(float, uy))
        uz = np.array(map(float, uz))
        vx = np.array(map(float, vx))
        vy = np.array(map(float, vy))
        vz = np.array(map(float, vz))
    
             
        
        
        #filter within xlim,ylim
        I=[]
        for i in range(len(x)):
            if (x[i]<=2.2 and x[i]>=-0.6 and y[i]<=0.6 and y[i]>=-0.6 ):
                I.append(i)
                
                
        x=x[I]
        y=y[I]
        ux=ux[I]
        uy=uy[I]
        uz=uz[I]
        vx=vx[I]
        vy=vy[I]
        vz=vz[I]
                 
     
        #plot
        def plot(xp,yp,zp,nc,name):
            plt.figure(figsize=(6, 5))
            #cp = pyplot.tricontour(ys, zs, pp,nc)
            cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
            plt.tricontourf(coord[ii][:,0],coord[ii][:,1],np.zeros(len(coord[ii])),colors='w')
            #cp=pyplot.tricontourf(x1,y1,z1)
            #cp=pyplot.tricontourf(x2,y2,z2)   
            
            #cp = pyplot.tripcolor(xp, yp, zp)
            #cp = pyplot.scatter(ys, zs, pp)
            #pyplot.clabel(cp, inline=False,fontsize=8)
            plt.xlim(-0.5,2)
            plt.ylim(-0.15,0.15)
            plt.colorbar(cp)
            #plt.axis('off')
            #plt.grid(True)
            #patch.set_facecolor('black')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.savefig('./plot_ts/%s.png'%(ii), format='png',bbox_inches='tight',dpi=200)
            plt.show()
            plt.close()
            
        plot(x,y,ux,20,'name')    
   
