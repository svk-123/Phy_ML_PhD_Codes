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
     
 
datafile='foil_image_216.pkl'
with open(datafile, 'rb') as infile:
    result = pickle.load(infile)
pname=result[0]   
para=result[1]
pname=np.asarray(pname)

aoa=[]
reno=[]
for i in range(len(tmp)):
    reno.append(tmp[i].split('_')[1])    
    aoa.append(tmp[i].split('_')[2])

reno=np.array(map(float, reno))
aoa = np.array(map(float, aoa))

#normalize
reno_max=5000
aoa_max=14
reno=reno/reno_max
aoa=aoa/aoa_max


img=[]
for i in range(len(foil)):
    if foil[i] in pname:
        ind=np.argwhere(pname==foil[i])
        img.append(para[int(ind)])

    else:
        print('not in pname %s'%foil[i])
img=np.asarray(img)

for i in range(len(img)):
    img[i][0,:]=reno[i]
    img[i][-1:,:]=reno[i]
    img[i][:,0]=reno[i]
    img[i][:,-1:]=reno[i]   
    img[i][1,:]=aoa[i]
    img[i][-2:-1,:]=aoa[i]
    img[i][:,1]=aoa[i]
    img[i][:,-2:-1]=aoa[i] 


cl=[]
cd=[]
cmm=[]
    
for ii in range(len(foil)):
    print ii
    casedir= path +'/%s/%s/postProcessing/forceCoeffs/0'%(foil[ii],tmp[ii])
                   
    xx=np.loadtxt(casedir +'/forceCoeffs.dat', skiprows=10)
    cmm.append(xx[-1:,1])
    cd.append(xx[-1:,2])
    cl.append(xx[-1:,3])        
             
        
info='[image, cl,cd,cm,re,aoa,name,info]'    
data2=[img,cl,cd,cmm,reno,aoa,foil,info]
with open('ml_input_output.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)        
        
        
        
        
        
        

        
