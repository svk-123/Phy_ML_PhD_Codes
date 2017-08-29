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

def calc_dns(uu):
    l=len(uu)
    UUDi=np.zeros((l,6))
    UUDi[:,0] =uu[:,0]
    UUDi[:,1] =uu[:,1]
    UUDi[:,2] =uu[:,2]
    UUDi[:,3] =uu[:,4]
    UUDi[:,4] =uu[:,5]
    UUDi[:,5] =uu[:,8]

    
    kD=0.5*(UUDi[:,0]+UUDi[:,3]+UUDi[:,5])
    
    # anisotropy tensor
    aD=np.zeros((l,6))
    
    aD[:,0]=UUDi[:,0]-(2./3.)*kD
    aD[:,1]=UUDi[:,1]
    aD[:,2]=UUDi[:,2]
    aD[:,3]=UUDi[:,3]-(2./3.)*kD
    aD[:,4]=UUDi[:,4]
    aD[:,5]=UUDi[:,5]-(2./3.)*kD
    
     #non dim. anisotropy tensor
    bD=np.zeros((l,6))  
    for i in range(l):
        bD[i,:]=aD[i,:]/(2*kD[i])
    
    
  
        
    return (UUDi,aD,bD)    
    
