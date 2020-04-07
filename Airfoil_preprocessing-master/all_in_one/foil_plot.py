#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Created on Mon Jul 17 22:49:29 2017

This code process OF data and exports as .pkl to prepData file
for TBNN. prepData reads .pkl and process further

@author: vino

"""

# imports
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import interpolate
from os import listdir
from os.path import isfile, isdir, join


#load file name
casedir='./picked_uiuc_71/'
fname = [f for f in listdir(casedir) if isfile(join(casedir, f))]
fname.sort()

nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])

'''
print ('writing..')

for i in range(len(fname)):
    
    xy1=np.genfromtxt('./foil_interp/%s.dat'%nname[i],skip_header=1,skip_footer=101)
    xy2=np.genfromtxt('./foil_interp/%s.dat'%nname[i],skip_header=102)
    
    fp=open('./foil_interp_reformat/%s.dat'%nname[i],'w')
    for j in range(99):
        fp.write('%f %f \n'%(xy1[j,0],xy1[j,1]))
    for j in range(100):
        fp.write('%f %f \n'%(xy2[j,0],xy2[j,1]))

    fp.close()        
'''


#checkplot
for i in range(len(fname)):    
    
    xy1=np.loadtxt('./picked_uiuc_101/%s.dat'%nname[i])
    #xy2=np.loadtxt('./foil_interp_reformat/%s.dat'%nname[i])
    #xy3=np.loadtxt('./coord_seligFmt_formatted/%s.dat'%nname[i],skiprows=1)
    
    #mei=int(len(xy3)/70)
    #if (mei==0):
    #    mei = 1
        
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(xy1[:,0],xy1[:,1],'r',label='MDO')
    #plt.plot(xy2[:,0],xy2[:,1],'g',label='PW')
    #plt.plot(xy3[:,0],xy3[:,1],'o',mfc='None',mew=1.0,mec='blue',ms=10,markevery=mei,label='CO')
    plt.legend()
        
    plt.ylim([-0.25,0.25])
    plt.savefig('./plot/%s.png'%nname[i],bbox_inches='tight',dpi=100)
    plt.show() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
