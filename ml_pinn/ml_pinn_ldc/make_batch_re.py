#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:49:13 2019

@author: vino
"""

"""
@author: Maziar Raissi
"""
'''
this is to make prediction using
p u v instead of original psi_p work
lamda removed
Re based training added
lr variable added
'''


import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import pickle

start_time = time.time()

np.random.seed(1234)
      
        
if __name__ == "__main__": 
      
           
    # Load Data
    #load data
    xtmp=[]
    ytmp=[]
    reytmp=[]
    utmp=[]
    vtmp=[]
    ptmp=[]
    
    relist=[100,200,400,600,1000,2000,4000,6000,8000,9000]
    #relist=[100,200,300,400,600,700,800,900]
    for ii in range(len(relist)):
        #x,y,Re,u,v
        with open('./data_file_st/cavity_Re%s.pkl'%relist[ii], 'rb') as infile:
            result = pickle.load(infile)
            
        N_train=100
    
        idx = np.random.choice(len(result[0]), N_train, replace=False)    
        xtmp.append(result[0][idx])
        ytmp.append(result[1][idx])
        reytmp.append(result[2][idx])
        utmp.append(result[3][idx])
        vtmp.append(result[4][idx])
        ptmp.append(result[5][idx])   
        
    xtmp=np.asarray(xtmp).transpose()
    ytmp=np.asarray(ytmp).transpose()
    utmp=np.asarray(utmp).transpose()
    vtmp=np.asarray(vtmp).transpose()
    ptmp=np.asarray(ptmp).transpose()
    reytmp=np.asarray(reytmp).transpose()/10000.    
       


    
    
    
  

             
    


