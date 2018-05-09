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

import keras
from keras.models import load_model
import shutil

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""

# duct - list
flist=['Re2000']
#flist=['Re1000']
Re=[2000]
#Re=[500]
AoA=range(0,11)

# read data from below dir...
path='./airfoil_post/n0012_ml/'

for ii in range(len(flist)):
    for jj in range(len(AoA)):
        
        #list dir & get max.time dir
        src=path+'%s/AoA_%s'%(flist[ii],AoA[jj])
        tmp=os.listdir(src)
        tmp1=[]
        for j in range(len(tmp)):
            if (str.isdigit(tmp[j]) == True):
                tmp1.append(tmp[j])
        tmp1 = map(int, tmp1)    
        sname=max(tmp1)
    
        dst=src+'/123'
        if not os.path.isdir(dst):
            os.makedirs(dst)
    
        x=[]
        with open(src+'/%s/ccx'%(sname), 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                x.append(line)
        x = np.array(map(float, x))
       
        y=[]
        with open(src+'/%s/ccy'%(sname), 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                y.append(line)
        y = np.array(map(float, y))
        
        z=[]
        with open(src+'/%s/ccz'%(sname), 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                z.append(line)
        z = np.array(map(float, z))
        
                #filter within xlim,ylim
        I=[]
        for i in range(len(x)):
            if (x[i]<=3.2 and x[i]>=-2.2 and y[i]<=2.2 and y[i]>=-2.2 ):
                I.append(i)
        xl=x[I]
        yl=y[I]
        zl=z[I]
        
        relist=[]
        for k in range(len(xl)):
            relist.append(Re[ii])
        relist=np.asarray(relist)    
        
 
        aoalist=[]
        for k in range(len(xl)):
            aoalist.append(AoA[jj])
        aoalist=np.asarray(aoalist) 
             
        #ml-predict        
        relist=relist/2000.
        aoalist=aoalist/11.
        val_inp=np.concatenate((xl[:,None],yl[:,None],relist[:,None],aoalist[:,None]),axis=1)
        
 
        model_test=load_model('./selected_model/final_sf.hdf5') 
        out=model_test.predict([val_inp])
        
        # p- write
        p=[]
        with open(src+'/%s/p'%(sname), 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                p.append(line)
        p = np.array(map(float, p))
        
        pl=p[I].copy()
        p[I]=out[:,2]
        
        print 'writing-p'
        fp= open(dst+'/p',"w+")
        
        for i in range(22):
            fp.write("%s"%(data0[i]))
        for i in range(npt):
            fp.write("%f\n"%(p[i]))
        for i in range((22+npt),len(data0)):    
            fp.write("%s"%(data0[i]))        
        fp.close() 
             
        #u-write
        u=[]
        v=[]
        w=[]
        with open(src+'/%s/U'%(sname), 'r') as infile:
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
        
        ul=u[I].copy
        vl=v[I].copy        
        wl=w[I].copy        
        
        u[I]=out[:,0]
        v[I]=out[:,1]
        
        print 'writing-u'
        fp= open(dst+'/U',"w+")
        
        for i in range(22):
            fp.write("%s"%(data0[i]))
        for i in range(npt):
            fp.write("(%f %f %f)\n"%(u[i],v[i],w[i]))
        for i in range((22+npt),len(data0)):    
            fp.write("%s"%(data0[i]))        
        fp.close() 
        
               
        #remove remaining dirs
        for kk in tmp1:
            if(kk!=0 and kk!=123):
                if os.path.exists(src+'/%s'%kk):
                    shutil.rmtree(src+'/%s'%kk)
                   
                
        #plot
        def plot(xp,yp,zp,nc,name):
            pyplot.figure(figsize=(6, 5), dpi=100)
            #cp = pyplot.tricontour(ys, zs, pp,nc)
            cp = pyplot.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
            #cp=pyplot.tricontourf(x1,y1,z1)
            #cp=pyplot.tricontourf(x2,y2,z2)   
            
            #cp = pyplot.tripcolor(xp, yp, zp)
            #cp = pyplot.scatter(ys, zs, pp)
            #pyplot.clabel(cp, inline=False,fontsize=8)
            pyplot.colorbar()
            pyplot.title(name)
            pyplot.xlabel('Z ')
            pyplot.ylabel('Y ')
            pyplot.xlim([-2,3])
            pyplot.ylim([-2,2])
            #pyplot.savefig(name, format='png', dpi=100)
            pyplot.show()
              
        #plot(xl,yl,pl,20,'cfd')    
        plot(x,y,p,20,'nn')           
        

        
