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
import matplotlib
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 


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

    cl_t=[]
    cl_p=[]
    cd_t=[]
    cd_p=[]
    
    for ii in range(5):
        print ii
        
        casedir= path +'/%s/%s/postProcessing/forceCoeffs1'%(foil[ii],tmp[ii])
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname.astype(np.int)
        ymax=int(yname.max())

                   
        xx1=np.loadtxt(casedir +'/%s/forceCoeffs.dat'%ymax, skiprows=10)
        xx1=xx1[-1:][0]
        
        casedir= path +'/%s/%s/postProcessing/forceCoeffs2'%(foil[ii],tmp[ii])        
        xx2=np.loadtxt(casedir +'/%s/forceCoeffs.dat'%ymax, skiprows=10)
        xx2=xx2[-1:][0]      
        
        cl_t.append(xx1[3] + xx2[3])
        cd_t.append(xx1[2] + xx2[2])   
        
             
        
        
        # lpred case dir
        casedirp= './foam_ml' + '/%s/%s_nn/postProcessing/forceCoeffs1'%(foil[ii],tmp[ii])
        #need to find max time later....
        yname = [f for f in listdir(casedirp) if isdir(join(casedirp, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname.astype(np.int)
        ymax=int(yname.max())
                   
        xx3=np.loadtxt(casedirp +'/%s/forceCoeffs.dat'%ymax, skiprows=10)
        xx3=xx3[-1:][0]
        
        casedirp= './foam_ml' + '/%s/%s_nn/postProcessing/forceCoeffs2'%(foil[ii],tmp[ii])       
        xx4=np.loadtxt(casedirp +'/%s/forceCoeffs.dat'%ymax, skiprows=10)
        xx4=xx4[-1:][0]       
        
        cl_p.append(xx3[3] + xx4[3])
        cd_p.append(xx3[2] + xx4[2])          
        
        
        
        
        
        
        

        