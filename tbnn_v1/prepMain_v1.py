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
path='../rans_data/data_r0/to_prepData/'

# call write file.txt
name='allData_r0_l7'    
#to read flist with suff.
suff='r0_l7'
write_file(flist,path,name,suff,True)        

def prep_input():
        
    #load written data
    k,ep,grad_u,stresses,coord=load_data(name)
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
with open('./datafile/to_ml/ml_%s.pkl'%name, 'wb') as outfile:
    pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


#plot
plot(coord[:,2],coord[:,1],tkedns,20,'tkedns')    
plot(coord[:,2],coord[:,1],k,20,'tkerans')        

# print time  
print("--- %s seconds ---" % (time.time() - start_time))
     
       