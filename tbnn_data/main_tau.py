#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:31:27 2017

@author: vino
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 19:35:45 2017

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

# time
import time
start_time = time.time()
                                 

# duct - list
#flist=['Re2200','Re2600','Re2900','Re3500']

flist=['Re3500']
# import functions
from calc_rans import calc_rans
from calc_dns import calc_dns
from plot_tau import plot,plotD

for ii in range(len(flist)):
    
    '''
    # import from sub routine
    from calc_dns import zD,yD,UUDi,aD,bD
    from tau_plot import plot,plotD
    from calc_rans import x,y,z,L,T1m,T2m,T3m,T4m,T5m,T6m
    from calc_rans import Tc,aR,bR,Rey,UUR
    from calc_rans import uu,uv,uw,vv,vw,ww
    from calc_rans import rxx,rxy,rxz,ryy,ryz,rzz
    from calc_rans import ux,uy,uz,vx,vy,vz,wx,wy,wz
    #meshgrid DNS co-ord for plotting
    ZD, YD = np.meshgrid(zD, yD, copy=False)
    
    '''
    x,z,y,UUR,Rey,aR,bR,L,T1m,T2m,T3m,T4m,T5m,T6m=calc_rans(flist[ii])
    
    UUDi,aD,bD=calc_dns(flist[ii],z,y)
    
    
    '''
    # plot -bij
    nbD=['uu-bD','uv-bD','uw-bD','vv-bD','vw-bD','ww-bD']
    nbR=['uu-bR','uv-bR','uw-bR','vv-bR','vw-bR','ww-bR']
    for i in range(6):
        plot(z,y,bD[:,i],20,'%s-%s'%(nbD[i],flist[ii]))
        plot(z,y,bR[:,i],20,'%s-%s'%(nbR[i],flist[ii]))
    '''
    
    
    '''
    # Plot-aij    
    naD=['uu-aD','uv-aD','uw-aD','vv-aD','vw-aD','ww-aD']
    naR=['uu-aR','uv-aR','uw-aR','vv-aR','vw-aR','ww-aR']
    for i in range(6):
        plot(z,y,aD[:,i],20,'%s-%s'%(naD[i],flist[ii]))
        plot(z,y,aR[:,i],20,'%s-%s'%(naR[i],flist[ii]))
    '''

    # Plot-Rxx from OF with DNS    
    nUUDi=['uu-Di','uv-Di','uw-Di','vv-Di','vw-Di','ww-Di']
    nUUR=['uu-UUR','uv-UUR','uw-UUR','vv-UUR','vw-UUR','ww-UUR']
    nRey=['uu-Rey','uv-Rey','uw-Rey','vv-Rey','vw-Rey','ww-Rey']
    for i in range(6):
        plot(z,y,UUDi[:,i],20,'%s_%s'%(nUUDi[i],flist[ii]))
        plot(z,y,UUR[:,i],20,'%s_%s'%(nUUR[i],flist[ii]))
        plot(z,y,Rey[:,i],20,'%s_%s'%(nRey[i],flist[ii]))


    '''
    # write data to ml
    data1 = [UUDi,bD,L,bR]
    with open('./data_out/ml_data1_%s.pkl'%flist[ii], 'wb') as outfile1:
        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)
    
    data2 = [T1m,T2m,T3m,T4m,T5m,T6m]
    with open('./data_out/ml_data2_%s.pkl'%flist[ii], 'wb') as outfile2:
        pickle.dump(data2, outfile2, pickle.HIGHEST_PROTOCOL)      
        
    data3 = [x,y,z]
    with open('./data_out/ml_data3_%s.pkl'%flist[ii], 'wb') as outfile3:
        pickle.dump(data3, outfile3, pickle.HIGHEST_PROTOCOL)     
    '''
        
    # print time  
    print("--- %s seconds ---" % (time.time() - start_time))
     
       