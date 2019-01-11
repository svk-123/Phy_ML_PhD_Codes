#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

@author: vinoth
"""

import time
start_time = time.time()

# Python 3.5
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from skimage import io, viewer,util 
from scipy import interpolate
import pandas as pd
from os import listdir
from os.path import isfile, join

import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)


path='./'

indir='./coord_seligFmt_formatted'

fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()  


nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])   

coord=[]
for i in range(10):
    print i
    coord.append(np.loadtxt(indir+'/%s.dat'%nname[i],skiprows=1))

xx=np.loadtxt('xx.txt')
foil_fp=[]

#plot
figure=plt.figure(figsize=(6,5))

f1=[0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5]
f2=[0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5]

for i in range(1):
    for j in range(10):
        print i
        l=len(coord[i])
        ind=np.argmin(coord[i][:,0])
        
        up_x=coord[i][:ind+1,0]
        up_y=coord[i][:ind+1,1]*f1[j]
        
        lr_x=coord[i][ind:,0]
        lr_y=coord[i][ind:,1]*f2[j]    
        
        up_x[0]=1
        up_x[-1:]=0
        
        lr_x[0]=0    
        lr_x[-1:]=1
        
        

        #plt0, =plt.plot(coord[i][:,0],coord[i][:,1],'o',linewidth=2,label='true')
        plt0, =plt.plot(up_x,up_y)
        plt0, =plt.plot(lr_x,lr_y)    
        #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
        #plt.legend(fontsize=16)
        #plt.xlabel('alpha',fontsize=16)
        #plt.ylabel('cl',fontsize=16)
        #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
        plt.xlim(0,1.)
        plt.ylim(-0.20,0.20)    
        plt.axis('off')
        #plt.savefig('./plotcheck/coord_%s.png'%nname[i], format='png')
        plt.show()
        plt.close()


    
    
    
    
    

