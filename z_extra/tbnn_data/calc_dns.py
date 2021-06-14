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

def calc_dns(name,z,y):
        
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


    ''' 
    Since, there is a confusion about dns z,y,
    vw of dns is multipled by -
    as to match with rans
    '''
    
    kD=0.5*(UUDi[:,0]+UUDi[:,3]+UUDi[:,5])
    
    # anisotropy tensor
    aD=np.zeros((len(z),6))
    
    aD[:,0]=UUDi[:,0]-(2./3.)*kD
    aD[:,1]=UUDi[:,1]
    aD[:,2]=UUDi[:,2]
    aD[:,3]=UUDi[:,3]-(2./3.)*kD
    aD[:,4]=UUDi[:,4]
    aD[:,5]=UUDi[:,5]-(2./3.)*kD
    
     #non dim. anisotropy tensor
    bD=np.zeros((len(z),6))  
    for i in range(len(z)):
        bD[i,:]=aD[i,:]/(2*kD[i])
    
    
    '''
    datauudi = [UUDi]
    with open('./data_out/UUDi_%s.pkl'%name, 'wb') as outfile4:
        pickle.dump(datauudi, outfile4, pickle.HIGHEST_PROTOCOL)
    '''    
   
        
    
        
    return (UUDi,aD,bD)    
    
