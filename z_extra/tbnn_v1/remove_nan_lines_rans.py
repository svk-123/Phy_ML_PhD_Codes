import os
import glob

import numpy as np
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate


path_r='../rans_data/wavywall/wavywall_Re6760_train_wnan.txt'

inan = np.loadtxt('wavywall_inan.txt', skiprows=0)
inan=inan+1

with open(path_r, 'r') as infile:
    data0=infile.readlines()

datan=[]    
for i in range(len(data0)):
    if i not in inan:
        datan.append(data0[i])
        
        

path_rn='../rans_data/wavywall/wavywall_Re6760_train.txt'
fp= open(path_rn,"w+")

for k in range(len(datan)):
    
    fp.write("%s"%datan[k])       
     
fp.close()         