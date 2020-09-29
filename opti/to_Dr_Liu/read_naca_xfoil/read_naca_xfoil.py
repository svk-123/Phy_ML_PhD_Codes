import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from skimage import io, viewer,util 
from scipy import interpolate
import pandas as pd
from os import listdir
from os.path import isfile, join
import os
import cPickle as pickle


#directory at which NACA airfoil results available
indir='./naca_xfoil/'

#output durectory
outdir='./outdir/'


#read all the files form the directory "indir"
#fname contains all the name of the files in "indir"
#fname.sort() sorts the names in alphabetical/acending order
fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort() 

#openfile 
fp=open(outdir + 'naca_output.txt','w')
fp.write('Airfoil, d1, d2, d3, Re, aoa, cl, cd, cm \n')

for i in range(len(fname)):
    
    #read airfoil name and split the digit
    # to get the airfoil name read 4th line in xfoil result file
    with open(indir + '%s'%fname[i], 'r') as infile:
        data=infile.readlines()
        
        # 4th line rwading
        tmp=data[3].strip()
        tmp=tmp.split('NACA')[1].strip()
        tmp=list(tmp)
        
        #extract the digits from the name
        d1=float(tmp[0])
        d2=float(tmp[1])
        d3=float(tmp[2]+tmp[3])
        
        #read Reynolds number from line 8
        tmp=data[8].split('Re')
        tmp=tmp[1].split('=')[1]
        tmp=tmp.split('Ncrit')[0].replace(' ','')
        Re=float(tmp)

        #read aoa, cl, dc, ect
        data2=np.loadtxt(indir + '%s'%fname[i],skiprows=12)
        
        #name
        name=fname[i].split('.')[0]
        
        #write details in file
        #name, d1, d2, d3, Re, AoA, Cl, cd, cm
        
        for j in range(len(data2)):
            fp.write('%s %f %f %f %f %f %f %f %f\n'%(name,d1,d2,d3,Re,\
                                                     data2[j,0],data2[j,1],data2[j,2],data2[j,4]))
            




fp.close()
        