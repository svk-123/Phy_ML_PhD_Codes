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
from prepData_v1 import get_rans_cbfs,get_dns_cbfs,plotust,write_file_cbfs,load_data

enforce_realizability = True
num_realizability_its = 5


case='duct'


#set path\
if(case=='wavywall'):
    path_r='../rans_data/wavywall/wavywall_Re6760_train.txt'
    path_d='../dns_data/wavywall/Re6760/wavywall_train.txt'

if(case=='hill'):
    path_r='../rans_data/hill/hill_Re10595_train_nn.txt'
    path_d='../dns_data/hill/Re10595/hill_train.dat'
    
if(case=='cbfs'):
    path_r='../rans_data/cbfs/cbfs_Re13700_train.txt'
    path_d='../dns_data/cbfs/Re13700/cbfs_train.dat'    
    
if(case=='duct'):
    path_r='../rans_data/duct/duct_Re3500_full.txt'
    Re='Re3500'       # only Re 
   
# call write file.txt
fname='duct_Re3500_full'    




if (case=='duct'):
    from prepData_v1 import get_rans_duct,get_dns_duct,write_file_duct


#cbfs-for_all_other_than_duct
if (case!='duct'): 
    write_file_cbfs(path_r,path_d,fname,case,True)  
if (case=='duct'):      
    write_file_duct(Re,path_r,fname,True)  

def prep_input():
        
    #load written data
    k,ep,grad_u,stresses,coord=load_data(fname)
    
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
    
    return(x,tb,y,coord,k,ep,tkedns,rans_bij)

x,tb,y,coord,k,ep,tkedns,rans_bij=prep_input()


# dump data
data=[x,tb,y,coord,k,ep,rans_bij,tkedns]
with open('./datafile/to_ml/ml_%s.pkl'%fname, 'wb') as outfile:
    pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


#plot
plotust(coord[:,2],coord[:,1],k,20,'tkedns')    
       

# print time  
print("--- %s seconds ---" % (time.time() - start_time))
     
       