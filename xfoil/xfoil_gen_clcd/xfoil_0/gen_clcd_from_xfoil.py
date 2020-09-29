"""
Optimize the airfoil shape directly using genetic algorithm, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)

Reference(s):
    Viswanath, A., J. Forrester, A. I., Keane, A. J. (2011). Dimension Reduction for Aerodynamic Design Optimization.
    AIAA Journal, 49(6), 1256-1266.
    Grey, Z. J., Constantine, P. G. (2018). Active subspaces of airfoil shape parameterizations.
    AIAA Journal, 56(5), 2003-2017.
"""

from __future__ import division
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

from simulation import evaluate,compute_coeff


indir='../picked_foil_0p5_parts/part_0/'

fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()  


Re=[10000,20000,40000,60000,80000,100000]
M=0
alpha=3
niter=200

for reno in range(1):
    for aoa in range(1):
        
        fp1=open('../foil_0p5_clcd/part_0/Re_%s_aoa_%s.dat'%(Re[0],alpha),'a')    
    
        for foil in range(1,2):
               
            print (foil)
            
            airfoil=np.loadtxt(indir + '%s'%fname[foil])
            
            a,b,c=evaluate(airfoil,Re[reno],0.0,alpha,niter)
            
            fp1.write('%s %f %f %f %f\n'%(fname[foil],Re[reno],alpha,b,c))
    
            
        fp1.close()



