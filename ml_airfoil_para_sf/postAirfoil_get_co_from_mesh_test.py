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
path='../cnn_airfoil_sf/OF_results/case_re_aoa_test_1'

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

    for ii in range(100):
        print ii
        
        casedir= path +'/%s/%s'%(foil[ii],tmp[ii])
                
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname[:-2].astype(np.int) 
        ymax=int(yname.max())
        
        xu=[]
        xl=[]
        with open(casedir +'/%s/Cx'%ymax, 'r') as infile:
            data0=infile.readlines()
            for k in range(len(data0)):
                if 'foil_upper' in data0[k]:
                    ind1=k
                if 'foil_lower' in data0[k]:
                    ind2=k       
                    
            n=int(data0[ind1+4]) 
            st=ind1+6
            for line in data0[st:st+n]:
                xu.append(line)
            xu = np.array(map(float, xu))
       
            n=int(data0[ind2+4]) 
            st=ind2+6
            for line in data0[st:st+n]:
                xl.append(line)
            xl = np.array(map(float, xl))        


        yu=[]
        yl=[]
        with open(casedir +'/%s/Cy'%ymax, 'r') as infile:
            data0=infile.readlines()
            for k in range(len(data0)):
                if 'foil_upper' in data0[k]:
                    ind1=k
                if 'foil_lower' in data0[k]:
                    ind2=k       
                    
            n=int(data0[ind1+4]) 
            st=ind1+6
            for line in data0[st:st+n]:
                yu.append(line)
            yu = np.array(map(float, yu))
       
            n=int(data0[ind2+4]) 
            st=ind2+6
            for line in data0[st:st+n]:
                yl.append(line)
            yl = np.array(map(float, yl))   








































    
