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
flist=['Re3500']

#flist=['Re3500']
# import functions
from calc_rans import calc_rans
from calc_dns import calc_dns
from plot_tau import plot,plotD

for ii in range(len(flist)):

    u00,u01,u02,u03,u04,u05,u06,u07,u08=calc_dns(flist[ii])
    k,ep,ux,uy,uz,vx,vy,vz,wx,wy,wz=calc_rans(flist[ii])    


#shuffle data
N= len(k)
I = np.arange(N)
np.random.shuffle(I)
n=1000 
k=k[I][:n]
ep=ep[I][:n]
ux=ux[I][:n]
uy=uy[I][:n]
uz=uz[I][:n]
vx=vx[I][:n]
vy=vy[I][:n]
vz=vz[I][:n]
wx=wx[I][:n]
wy=wy[I][:n]
wz=wz[I][:n]

u00=u00[I][:n]
u01=u01[I][:n]
u02=u02[I][:n]
u03=u03[I][:n]
u04=u04[I][:n]
u05=u05[I][:n]
u06=u06[I][:n]
u07=u07[I][:n]
u08=u08[I][:n]


   

    
    
print("--- %s seconds ---" % (time.time() - start_time))

l=len(k)     

print 'writing..'
fp= open("to_tbnn.txt","w+")

fp.write('line to skip\n')
fp.write('line to skip\n')
fp.write('line to skip\n')
fp.write('line to skip\n')

for i in range(l):
    fp.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n"\
             %(k[i],ep[i],ux[i],uy[i],uz[i],vx[i],vy[i],vz[i],wx[i],wy[i],wz[i],u00[i],u01[i],u02[i],u03[i],u04[i],u05[i],u06[i],u07[i],u08[i]))        
    
fp.close()
    