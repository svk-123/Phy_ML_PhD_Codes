#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:22:23 2017

@author: vino
"""
# imports
import os
import glob

import numpy as np
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate
import cPickle as pickle

def calc_dns(name):
        
    #read DNS data
    dataframe = pd.read_csv('../dns_data/duct/z_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    zD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/y_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    yD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/um_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    uD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/vm_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    vD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/wm_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    wD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/uu_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    uuD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/uv_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    uvD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/uw_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    uwD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/vv_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    vvD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/vw_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    vwD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/ww_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    wwD=np.asarray(dataset)
    
    '''
    #----- commment St-----------------#
    # calculations: no change required
    Z, Y = np.meshgrid(zD, yD, copy=False) 
    
    fuuD = interpolate.interp2d(Z, Y, uuD, kind='linear')
    fuvD = interpolate.interp2d(Z, Y, uvD, kind='linear')
    fuwD = interpolate.interp2d(Z, Y, uwD, kind='linear')
    fvvD = interpolate.interp2d(Z, Y, vvD, kind='linear')
    fvwD = interpolate.interp2d(Z, Y, vwD, kind='linear')
    fwwD = interpolate.interp2d(Z, Y, wwD, kind='linear')
    
    #from calc_rans import x,y,z
    #interpolate DNS to rans co-ordinates
    UUDi=np.zeros((len(z),6))
    for i in range(len(z)):
        UUDi[i,0]=fuuD(z[i],y[i])
        UUDi[i,1]=fuvD(z[i],y[i])
        UUDi[i,2]=fuwD(z[i],y[i])
        UUDi[i,3]=fvvD(z[i],y[i])
        UUDi[i,4]=fvwD(z[i],y[i])
        UUDi[i,5]=fwwD(z[i],y[i])
    #--------comment END---------------#
    '''
    

    with open('./data_out/UUDi_%s.pkl'%name, 'rb') as infile4:
        result4 = pickle.load(infile4)
    UUDi=result4[0]

    uuDNS=np.zeros((len(UUDi),9))
    uuDNS[:,0]=UUDi[:,0]
    uuDNS[:,1]=UUDi[:,1]
    uuDNS[:,2]=UUDi[:,2]
    uuDNS[:,3]=UUDi[:,1]
    uuDNS[:,4]=UUDi[:,3]
    uuDNS[:,5]=UUDi[:,4]
    uuDNS[:,6]=UUDi[:,2]
    uuDNS[:,7]=UUDi[:,4]
    uuDNS[:,8]=UUDi[:,5]
  
    return(uuDNS[:,0],uuDNS[:,1],uuDNS[:,2],uuDNS[:,3],uuDNS[:,4],uuDNS[:,5],uuDNS[:,6],uuDNS[:,7],uuDNS[:,8])
    
