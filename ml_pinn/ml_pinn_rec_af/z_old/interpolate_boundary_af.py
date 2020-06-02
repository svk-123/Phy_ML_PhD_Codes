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


fname_1=['naca0012']

fname_1=np.asarray(fname_1)
fname_1.sort()


# read data from below dir...
path='./foam_case/'
#path='/home/vino/ml_test/from_nscc_26_dec_2018/foam_run/case_naca_lam'
indir = path



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

       
coord=[]
#for nn in range(len(foil)):
#    pts=np.loadtxt('./coord_seligFmt_formatted/%s.dat'%foil[nn],skiprows=1)
#    coord.append(pts)
# 


aoa=[]
reno=[]
for i in range(len(tmp)):
    reno.append(tmp[i].split('_')[1])    
    aoa.append(tmp[i].split('_')[2])

reno=np.array(map(float, reno))
aoa = np.array(map(float, aoa))


st= [0]
end=[1]


'''np.random.seed(1234534)
mylist=np.random.randint(0,4500,100)

nco=[]
for k in mylist:
    nco.append(coord[k])'''

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

    #for ii in mylist:
    for ii in range(st[jj],end[jj]):
        print (ii)
        
        casedir= path +'%s/%s'%(foil[ii],tmp[ii])
                
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname[:-3].astype(np.int) 
        ymax=int(yname.max())



        x=[]
        with open(casedir +'/%s/ccx'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                x.append(line)
        x = np.array(map(float, x))
       
        y=[]
        with open(casedir +'/%s/ccy'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                y.append(line)
        y = np.array(map(float, y))
        
        z=[]
        with open(casedir +'/%s/ccz'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                z.append(line)
        z = np.array(map(float, z))
        
        p=[]
        with open(casedir +'/%s/p'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                p.append(line)
        p = np.array(map(float, p))
        
        
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
               
        ab=np.loadtxt('./data_file/af_inout_sq_5556_4s_200.dat')
        
      
        #LinearNDinterpolator
        pD=np.asarray([x,y]).transpose()
    
        #for -u
        print 'interpolation-1...'      
        f1u=interpolate.LinearNDInterpolator(pD,u)
        bu=np.zeros(len(ab))
        for j in range(len(ab)):
            bu[j]=f1u(ab[j,0],ab[j,1])
             
        #for -v
        print 'interpolation-1...'      
        f1v=interpolate.LinearNDInterpolator(pD,v)
        bv=np.zeros(len(ab))
        for j in range(len(ab)):
            bv[j]=f1v(ab[j,0],ab[j,1])

        #for -p
        print 'interpolation-1...'      
        f1p=interpolate.LinearNDInterpolator(pD,p)
        bp=np.zeros(len(ab))
        for j in range(len(ab)):
            bp[j]=f1p(ab[j,0],ab[j,1])
            
            
        fp=open('./data_file/af_inout_sq_5556_4s_200_intp.dat','w')

        for i in range(len(ab)):
            fp.write('%f %f %f %f %f\n'%(ab[i,0],ab[i,1],bu[i],bv[i],bp[i]))

        fp.close()        
            