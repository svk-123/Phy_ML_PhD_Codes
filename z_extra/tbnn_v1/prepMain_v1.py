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
                                 
from turbulencekepspreprocessor_v1 import TurbulenceKEpsDataProcessor
tdp=TurbulenceKEpsDataProcessor()
from prepData_v1 import get_rans_cbfs,get_dns_cbfs,plotust,write_file_cbfs,load_data,load_data_piml

enforce_realizability = True
num_realizability_its = 5


case='duct'


#set path
if(case=='wavywall'):
    path_r='../rans_data/wavywall/wavywall_Re6760_full.txt'
    path_d='../dns_data/wavywall/Re6760/wavywall_train.txt'

if(case=='hill'):
    path_r='../rans_data/hill/hill_Re10595_full.txt'
    path_d='../dns_data/hill/Re10595/hill_train.dat'
    
if(case=='cbfs'):
    path_r='../rans_data/cbfs/cbfs_Re13700_train.txt'
    path_d='../dns_data/cbfs/Re13700/cbfs_train.dat'    
    
if(case=='duct'):
    Re='Re3500'       # only Re 
    path_r='../rans_data/duct/duct_%s_full.txt'%Re

   
# call write file.txt
#fname='wavywall_Re6760_train'    
fname='duct_%s_full'%Re # do not include.txt


if (case=='duct'):
    from prepData_v1 import get_rans_duct,get_dns_duct,write_file_duct

#cbfs-for_all_other_than_duct
if (case!='duct'): 
    write_file_cbfs(path_r,path_d,fname,case,True)  
if (case=='duct'):      
    write_file_duct(Re,path_r,fname,True)  


def prep_input():
        
    #load written data
    k,ep,grad_u,stresses,coord,wd=load_data(fname)
    
    # Calculate inputs and outputs
    Sij, Rij = tdp.calc_Sij_Rij(grad_u, k, ep)
    x = tdp.calc_scalar_basis(Sij, Rij)  # Scalar basis
    tb = tdp.calc_tensor_basis(Sij, Rij)  # Tensor basis
    y,tkedns = tdp.calc_output(stresses)  # Anisotropy tensor
    rans_bij=tdp.calc_rans_anisotropy(grad_u, k, ep)
    
    # Enforce realizability
    if enforce_realizability:
        for i in range(num_realizability_its):
            y = tdp.make_realizable(y)
    
    #tke input
    I = tdp.calc_tke_input(Sij, Rij)
    
    return(x,tb,y,coord,k,ep,tkedns,rans_bij,I,grad_u,wd)

x,tb,y,coord,k,ep,tkedns,rans_bij,I,grad_u,wd=prep_input()


def prep_piml_input():
       
    #piml
    k,ep,grad_u,grad_p,grad_k,vel,coord=load_data_piml(fname)
    B=tdp.calc_piml_basis(k,ep,grad_u,grad_p,grad_k,vel)

    return B

B = prep_piml_input()
    
# dump data
wd=abs(wd)
data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I,B,wd]

l1=[]
l2=[]
for i in range(len(wd)):
    if (wd[i]<=0.1):
        l1.append(i)
    else:
        l2.append(i)

x1          = x[l1][:]
tb1         = tb[l1][:]
y1          = y[l1][:]
c1          = coord[l1][:]
k1          = k[l1][:]
ep1         = ep[l1][:]
rans_bij1   = rans_bij[l1][:]
tkedns1     = tkedns[l1][:]
I1          = I[l1][:]
B1          = B[l1][:]
wd1         = wd[l1][:]

x2          = x[l2][:]
tb2         = tb[l2][:]
y2          = y[l2][:]
c2          = coord[l2][:]
k2          = k[l2][:]
ep2         = ep[l2][:]
rans_bij2   = rans_bij[l2][:]
tkedns2     = tkedns[l2][:]
I2          = I[l2][:]
B2          = B[l2][:]
wd2         = wd[l2][:]


data1=[x1,tb1,y1,c1,k1,ep1,rans_bij1,tkedns1,I1,B1,wd1]
with open('./datafile/to_ml/1_ml_%s.pkl'%fname, 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)
    
data2=[x2,tb2,y2,c2,k2,ep2,rans_bij2,tkedns2,I2,B2,wd2]
with open('./datafile/to_ml/2_ml_%s.pkl'%fname, 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)  
    
#plot
for j in range(6):
    plotust(coord[:,2],coord[:,1],y[:,j],20,'dns')    
    plotust(coord[:,2],coord[:,1],rans_bij[:,j],20,'rans')          
plotust(coord[:,2],coord[:,1],wd,20,'rans')  
'''
for i in range(3):
    for j in range(3):    
        plotust(coord[:,2],coord[:,1],grad_u[:,i,j],20,'dns')    
'''
 
    
    
    
# print time  
print("--- %s seconds ---" % (time.time() - start_time))
     
       
