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

indir='./picked_uiuc_101'

fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()  


nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])   

coord=[]
for i in range(len(fname)):
    print i
    coord.append(np.loadtxt(indir+'/%s.dat'%nname[i]))

#xx=np.loadtxt('xx.txt')
#foil_fp=[]

#plot
figure=plt.figure(figsize=(6,5))

f1=np.asarray([0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.05,1.1,1.15,1.2,1.25,1.3,1.35])
f2=f1.copy()
np.random.seed(36284)



for i in range(1433):
    np.random.shuffle(f1)
    np.random.shuffle(f2)
    
#    for l in range(f1):
#        if((f1[l]-f2[l]) <0.2):
#             pass   
    
    for j in range(10):
        print ('step:',i,j)
        l=len(coord[i])
        ind=np.argmin(coord[i][:,0])
        coord[i][:,1]=coord[i][:,1]-coord[i][ind,1]
        #first term not scaled 
        up_x=coord[i][:ind+1,0]
        up_y=coord[i][:ind+1,1]
        lr_x=coord[i][ind:,0]
        lr_y=coord[i][ind:,1]  
        
        up_y[1:-1]=up_y[1:-1]*f1[j]
        lr_y[1:-1]=lr_y[1:-1]*f2[j]
        
        dist1=up_y[10]-lr_y[-11]
        dist2=up_y.max()
        
        print(dist1,dist2)  
        
        if(dist1 > 0.005 and dist2 < 0.18):
            #plt0, =plt.plot(coord[i][:,0],coord[i][:,1],'o',linewidth=2,label='true')
            plt0, =plt.plot(up_x,up_y)
            plt0, =plt.plot(lr_x,lr_y)    
            #plot0,=plt.plot([-0.05,1.05],[0,0],'k')
            plot0,=plt.plot([])
            #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
            #plt.legend(fontsize=16)
            #plt.xlabel('alpha',fontsize=16)
            #plt.ylabel('cl',fontsize=16)
            #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
            #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
            plt.xlim(-0.05,1.05)
            plt.ylim(-0.25,0.25)    
            plt.axis('off')
            plt.savefig('./plotcheck/%s_%d.png'%(nname[i],j), format='png')
            plt.show()
            plt.close()
    
            fp=open('./picked_uiuc_aug/%s_%d.dat'%(nname[i],j),'w+')
            fp.write('%s_%d\n'%(nname[i],j))
            for l in range(len(up_x)):
                fp.write('%f %f \n'%(up_x[l],up_y[l]))
            for l in range(len(lr_x)-1):
                fp.write('%f %f \n'%(lr_x[l+1],lr_y[l+1]))  
        
            fp.close()
    
