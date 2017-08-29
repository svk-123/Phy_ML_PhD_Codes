#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:51:09 2017

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

def calc_rans(name):
    
    # read RANS data
    # ref:[x,y,z,u,v,w,k,ep,nut]
    with open('../rans_data/duct_rans_data1_%s.pkl'%name, 'rb') as infile1:
        result1 = pickle.load(infile1)
        
    #ref:data2 = [ux,uy,uz,vx,vy,vz,wx,wy,wz]
    with open('../rans_data/duct_rans_data2_%s.pkl'%name, 'rb') as infile2:
        result2 = pickle.load(infile2)
    
    #ref:data3 = [rxx,rxy,rxz,ryy,ryz,rzz]
    with open('../rans_data/duct_rans_data3_%s.pkl'%name, 'rb') as infile3:
        result3 = pickle.load(infile3)
        
    x,y,z=result1[0],result1[1],result1[2]
    u,v,w=result1[3],result1[4],result1[5]
    k,ep,mut=result1[6],result1[7],result1[8]
    
    ux,uy,uz=result2[0],result2[1],result2[2]
    vx,vy,vz=result2[3],result2[4],result2[5]
    wx,wy,wz=result2[6],result2[7],result2[8]
    
    rxx,rxy,rxz=result3[0],result3[1],result3[2]
    ryy,ryz,rzz=result3[3],result3[4],result3[5]
    

    return (k,ep,ux,uy,uz,ux,uy,uz,wx,wy,wz)
  
    
    
    
    
    
    
    
    
