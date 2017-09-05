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
from prepData_v1 import get_rans,get_dns,plot,plotD,write_file,load_data

# list of data
flist=['Re2200','Re2600','Re2900','Re3500']
#flist=['Re3500']
enforce_realizability = True
num_realizability_its = 5
path='../rans_data/data_r1/to_prepData_l1/'

# call write file
name='allDataQ_l1'    
write_file(flist,path,name)        

def prep_input():
        
    #load written data
    k,ep,grad_u,stresses,coord=load_data(name)
    # Calculate inputs and outputs
    Sij, Rij = tdp.calc_Sij_Rij(grad_u, k, ep)
    x = tdp.calc_scalar_basis(Sij, Rij)  # Scalar basis
    tb = tdp.calc_tensor_basis(Sij, Rij)  # Tensor basis
    y = tdp.calc_output(stresses)  # Anisotropy tensor
    
    # Enforce realizability
    if enforce_realizability:
        for i in range(num_realizability_its):
            y = tdp.make_realizable(y)
    
    return(x,tb,y,coord,k,ep)


x,tb,y,coord,k,ep=prep_input()

data=[x,tb,y,coord,k,ep]
with open('./datafile/to_ml/ml_%s.pkl'%name, 'wb') as outfile:
    pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
     
    
       
# print time  
print("--- %s seconds ---" % (time.time() - start_time))
     
       