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


import keras
from keras.models import load_model
import shutil

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


#no use loop
for jj in range(1):

    for ii in range(2):
        print ii
        
        casedir= path +'/%s/%s'%(foil[ii],tmp[ii])
                
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname[:-3].astype(np.int) 
        ymax=int(yname.max())
        
        xu=[]
        with open(casedir +'/%s/Cx'%ymax, 'r') as infile:
            data0=infile.readlines()
            
            ln=0
            for line in data0:
                if 'foil_upper' in line:
                    idx=ln
                ln=ln+1    
            num=int(data0[idx+4])
            
            for line in data0[idx+6:idx+6+num]:
                xu.append(line)
        xu = np.array(map(float, xu)) 
       
        xl=[]
        with open(casedir +'/%s/Cx'%ymax, 'r') as infile:
            data0=infile.readlines()
            
            ln=0
            for line in data0:
                if 'foil_lower' in line:
                    idx=ln
                ln=ln+1    
            num=int(data0[idx+4])
            
            for line in data0[idx+6:idx+6+num]:
                xl.append(line)
        xl = np.array(map(float, xl))        

        yu=[]
        with open(casedir +'/%s/Cx'%ymax, 'r') as infile:
            data0=infile.readlines()
            
            ln=0
            for line in data0:
                if 'foil_upper' in line:
                    idx=ln
                ln=ln+1    
            num=int(data0[idx+4])
            
            for line in data0[idx+6:idx+6+num]:
                yu.append(line)
        yu = np.array(map(float, yu)) 
       
        yl=[]
        with open(casedir +'/%s/Cx'%ymax, 'r') as infile:
            data0=infile.readlines()
            
            ln=0
            for line in data0:
                if 'foil_lower' in line:
                    idx=ln
                ln=ln+1    
            num=int(data0[idx+4])
            
            for line in data0[idx+6:idx+6+num]:
                yl.append(line)
        yl = np.array(map(float, yl))          
       
                       
        # load shearstress - true upper
        tu1=[]
        tu2=[]
        with open(casedir +'/%s/wallShearStress'%ymax, 'r') as infile:
            data0=infile.readlines()
            ln=0
            for line in data0:
                if 'foil_upper' in line:
                    idx=ln
                ln=ln+1    
            num=int(data0[idx+4])
            
            for line in data0[idx+6:idx+6+num]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c = (item.strip() for item in line.split(' ', 3))
                tu1.append(a), tu2.append(b)
        tu1 = np.array(map(float, tu1))
        tu2 = np.array(map(float, tu2))
        tu3=np.sqrt(tu1**2 + tu2**2)
        
        # load shearstress - true upper
        tl1=[]
        tl2=[]
        with open(casedir +'/%s/wallShearStress'%ymax, 'r') as infile:
            data0=infile.readlines()
            ln=0
            for line in data0:
                if 'foil_lower' in line:
                    idx=ln
                ln=ln+1    
            num=int(data0[idx+4])
            
            for line in data0[idx+6:idx+6+num]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c = (item.strip() for item in line.split(' ', 3))
                tl1.append(a), tl2.append(b)
        tl1 = np.array(map(float, tl1))
        tl2 = np.array(map(float, tl2))
        tl3=np.sqrt(tl1**2 + tl2**2)        
                       
        
        # lpred case dir
        casedirp= './foam_ml' +'/%s/%s_nn'%(foil[ii],tmp[ii])

        # load shearstress - pred upper
        tu1p=[]
        tu2p=[]
        with open(casedirp +'/%s/wallShearStress'%ymax, 'r') as infile:
            data0=infile.readlines()
            ln=0
            for line in data0:
                if 'foil_upper' in line:
                    idx=ln
                ln=ln+1    
            num=int(data0[idx+4])
            
            for line in data0[idx+6:idx+6+num]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c = (item.strip() for item in line.split(' ', 3))
                tu1p.append(a), tu2p.append(b)
        tu1p = np.array(map(float, tu1p))
        tu2p = np.array(map(float, tu2p))    
        tu3p=np.sqrt(tu1p**2 + tu2p**2)

        # load shearstress - pred upper
        tl1p=[]
        tl2p=[]
        with open(casedirp +'/%s/wallShearStress'%ymax, 'r') as infile:
            data0=infile.readlines()
            ln=0
            for line in data0:
                if 'foil_lower' in line:
                    idx=ln
                ln=ln+1    
            num=int(data0[idx+4])
            
            for line in data0[idx+6:idx+6+num]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c = (item.strip() for item in line.split(' ', 3))
                tl1p.append(a), tl2p.append(b)
        tl1p = np.array(map(float, tl1p))
        tl2p = np.array(map(float, tl2p))    
        tl3p=np.sqrt(tl1p**2 + tl2p**2)


        #figure
        plt.figure(figsize=(6, 4))
        a0=plt.plot(xu,tu3,label='true')
        a1=plt.plot(xu,tu3p,label='nn')
        plt.xlim([0,1])
        plt.legend()
        plt.show()
        
        
        
        
        
        