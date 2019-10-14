#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
reads cl, cd, cm from post processing folder

@author: vino
"""
# imports
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import interpolate
from os import listdir
from os.path import isfile,isdir, join
import cPickle as pickle
import shutil

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""

# read data from below dir...
path='./case'

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

aoa=[]
reno=[]
for i in range(len(tmp)):
    reno.append(tmp[i].split('_')[1])    
    aoa.append(tmp[i].split('_')[2])

reno=np.array(map(float, reno))
aoa = np.array(map(float, aoa))



para=[]
#split space from name
for i in range(len(foil)):
    tmp0=foil[i].split('naca')[1].strip()
    tmp1=list(tmp0)
    d1=float(tmp1[0])
    d2=float(tmp1[1])
    d3=float(tmp1[2]+tmp1[3])
    
    para.append([d1,d2,d3])
    
para=np.asarray(para)



#np.random.seed(1234534)
#mylist=np.random.randint(0,450,20)

#no use loop
cl=[]
cd=[]
cm=[]


for jj in range(1):

    for ii in range(0,len(foil)):
        print ii
        
        casedir= path +'/%s/%s/postProcessing/forceCoeffs'%(foil[ii],tmp[ii])
                
        xx=np.loadtxt(casedir +'/0/forceCoeffs.dat', skiprows=10)

        if (len(xx) < 1998):
            xx1=xx[-1:][0] 
        
        else:
            xx1= sum(xx[-200:])/len(xx[-200:])

        cm.append(xx1[1])
        cd.append(xx1[2])
        cl.append(xx1[3])

cm=np.asarray(cm)
cd=np.asarray(cd)
cl=np.asarray(cl)
      

##save file
#filepath='./data_file'
#      
## ref:[x,y,z,ux,uy,uz,k,ep,nut]
#info= '[cm, cd, cl, reno, aoa, para, name, info ]'
#
#data1 = [cm, cd, cl, reno, aoa, para, foil, info ]
#
#with open(filepath+'/naca4_clcd_lam_1.pkl', 'wb') as outfile1:
#    pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)
    

fp = open('foam_val.dat','w+')
fp.write("name   Reno   AoA   Cm   Cd   Cl\n\n")
for i in range(len(cl)):
    fp.write("%s   %d   %d   %0.4f  %0.4f  %0.4f \n\n"%(foil[i],reno[i],aoa[i],cm[i],cd[i],cl[i]))
    
fp.close()    





      
