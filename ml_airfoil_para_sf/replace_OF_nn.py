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
        
        #filter within xlim,ylim
        I=[]
        for i in range(len(x)):
            if (x[i]<=2 and x[i]>=-0.5 and y[i]<=0.5 and y[i]>=-0.5 ):
                I.append(i)
        xl=x[I]
        yl=y[I]
        zl=z[I]
        
        relist=[]
        for k in range(len(xl)):
            relist.append(reno[ii])
        relist=np.asarray(relist)    
        
 
        aoalist=[]
        for k in range(len(xl)):
            aoalist.append(aoa[jj])
        aoalist=np.asarray(aoalist) 
        
        paralist=[]
        for k in range(len(xl)):
            paralist.append(my_para[jj])
        paralist=np.asarray(paralist) 
             
        #ml-predict        
        relist=relist/2000.
        aoalist=aoalist/14.
        val_inp=np.concatenate((xl[:,None],yl[:,None],relist[:,None],aoalist[:,None],paralist[:,:]),axis=1)
        
 
        model_test=load_model('./selected_model/case_9_naca_lam_1/model_sf_65_0.00000317_0.00000323.hdf5') 
        out=model_test.predict([val_inp])
        
        # p- write
        p=[]
        with open(casedir +'/%s/p'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                p.append(line)
        p = np.array(map(float, p))
        
        pl=p[I].copy()
        p[I]=out[:,0]
        
                
        dst2='./foam_ml' +'/%s/%s_nn'%(foil[ii],tmp[ii])
        
        if os.path.exists(dst2):
            shutil.rmtree(dst2)
                    
        shutil.copytree(casedir,dst2)
        
        print 'writing-p'
        fp= open(dst2 +'/%s/p'%ymax, 'w+')
        
        for i in range(22):
            fp.write("%s"%(data0[i]))
        for i in range(npt):
            fp.write("%f\n"%(p[i]))
        for i in range((22+npt),len(data0)):    
            fp.write("%s"%(data0[i]))        
        fp.close() 
             

        # load velocity
        u=[]
        v=[]
        with open(casedir +'/%s/U'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c = (item.strip() for item in line.split(' ', 3))
                u.append(a), v.append(b)
        u = np.array(map(float, u))
        v = np.array(map(float, v))

        u[I]=out[:,1]
        v[I]=out[:,2]                      
        

        print 'writing-U'
        fp= open(dst2 +'/%s/U'%ymax, 'w+')
        
        for i in range(22):
            fp.write("%s"%(data0[i]))
        for i in range(npt):
            fp.write("(%f %f 0.0)\n"%(u[i],v[i]))
        for i in range((22+npt),len(data0)):    
            fp.write("%s"%(data0[i]))        
        fp.close()         
   
        


    
      
