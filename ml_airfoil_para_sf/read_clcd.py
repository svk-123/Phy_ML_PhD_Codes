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
path='./paper_plots/clcd/foam'

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
fname_2=np.asarray(fname_2)
fname_2.sort()
    
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

    cl_t1=[]
    cl_p1=[]
    cd_t1=[]
 
    cl_t2=[]
    cl_p2=[]
    cd_t2=[]

    cm_t1=[]
    cm_t2=[]
    
    
    for ii in range(6):
        print ii
        ## Time       	Cm           	Cd           	Cl           	Cl(f)        	Cl(r) 
        casedir= path +'/%s/%s/postProcessing/forceCoeffs1'%(foil[ii],tmp[ii])
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname.astype(np.int)
        ymax=int(yname.max())
                   
        xx1=np.loadtxt(casedir +'/%s/forceCoeffs.dat'%ymax, skiprows=10)
        xx1=xx1[-1:][0]
        cd_t1.append(xx1[2])
        cl_t1.append(xx1[3])          
        cl_p1.append(xx1[5])
        cm_t1.append(xx1[1])
                
        
        # lpred case dir
        casedirp= './paper_plots/clcd/ml' + '/%s/%s_nn/postProcessing/forceCoeffs1'%(foil[ii],tmp[ii])
        #need to find max time later....
        yname = [f for f in listdir(casedirp) if isdir(join(casedirp, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname.astype(np.int)
        ymax=int(yname.max())
                   
        xx2=np.loadtxt(casedirp +'/%s/forceCoeffs.dat'%ymax, skiprows=10)
        xx2=xx2[-1:][0]
        
        cd_t2.append(xx2[2])
        cl_t2.append(xx2[3])          
        cl_p2.append(xx2[5])  
        cm_t2.append(xx2[1])        
         
        #sort_by_aoa
        #cl 1, cl2,cd_1,cd_2,aoa
        
        
    plt.figure(figsize=(6, 6), dpi=100)
        
    plt0, =plt.plot(aoa,cl_t1,'o',mfc='b',mew=1.5,mec='b',ms=10,label='CFD-$C_l$')
    plt0, =plt.plot(aoa,cd_t1,'s',mfc='b',mew=1.5,mec='b',ms=10,label='CFD-$C_d$') 
    plt0, =plt.plot(aoa,cm_t1,'^',mfc='b',mew=1.5,mec='b',ms=10,label='CFD-$C_m$') 
                
    plt0, =plt.plot(aoa,cl_t2,'r',lw=3,marker='o',mfc='None',mew=1.5,mec='r',ms=10,label='MLP-$C_l$')
    plt0, =plt.plot(aoa,cd_t2,'r',lw=3,marker='s',mfc='None',mew=1.5,mec='r',ms=10,label='MLP-$C_d$')     
    plt0, =plt.plot(aoa,cm_t2,'r',lw=3,marker='^',mfc='None',mew=1.5,mec='r',ms=10,label='MLP-$C_m$')  
  
    

    plt.xlabel('AoA',fontsize=20)
    plt.ylabel('$C_l$,$C_d$,$C_m$' ,fontsize=20)
    #plt.title('%s-AoA-%s-p'%(flist[ii],AoA[jj]),fontsize=16)
    plt.subplots_adjust(top = 0.95, bottom = 0.2, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
    plt.figtext(0.45, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)
    plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(-0.0,0.75), ncol=1, frameon=False, fancybox=False, shadow=False)
    plt.xlim(-0.3,15)
    plt.ylim(-0.01,0.5) 
    plt.xticks([0,4,8,12])
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.savefig('./plot_ts/clcd_2.tiff', format='tiff',bbox_inches='tight', dpi=300)
    plt.show()    
    plt.close()        
        
        
        

        
