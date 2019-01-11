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
for i in range(len(fname)):
    print i
    coord.append(np.loadtxt(indir+'/%s.dat'%nname[i],skiprows=1))


xx=np.loadtxt('xx.txt')
foil_fp=[]
for iiii in range(1):
    for i in range(len(nname)):
        print i
        l=len(coord[i])
        ind=np.argmin(coord[i][:,0])
        
        up_x=coord[i][:ind+1,0]
        up_y=coord[i][:ind+1,1]
        
        lr_x=coord[i][ind:,0]
        lr_y=coord[i][ind:,1]    
        
        up_x[0]=1
        up_x[-1:]=0
        
        lr_x[0]=0    
        lr_x[-1:]=1
        
        fu = interpolate.interp1d(up_x, up_y)
        u_yy = fu(xx)
        
        fl = interpolate.interp1d(lr_x, lr_y)
        l_yy = fl(xx)   
                
        yout=np.zeros(len(u_yy)*2)
        yout[0:len(xx)]=u_yy
        yout[len(xx):]=l_yy    
        foil_fp.append(yout)
        
        #plot
        figure=plt.figure(figsize=(6,5))
        plt0, =plt.plot(coord[i][:,0],coord[i][:,1],'o',linewidth=2,label='true')
        plt0, =plt.plot(xx,u_yy)
        plt0, =plt.plot(xx,l_yy)    
        #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
        #plt.legend(fontsize=16)
        #plt.xlabel('alpha',fontsize=16)
        #plt.ylabel('cl',fontsize=16)
        #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
        plt.xlim(0,1.)
        plt.ylim(-0.20,0.20)    
        plt.axis('off')
        plt.savefig('./plotcheck/coord_%s.png'%nname[i], format='png')
        plt.close()

     
    
 
foil_mat=[]
for iiii in range(1):
    for i in range(len(nname)):
        print i
        #plot
        figure=plt.figure(figsize=(3,3))
        plt0, =plt.plot(coord[i][:,0],coord[i][:,1],'k',linewidth=0.1,label='true')
        #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
        #plt.legend(fontsize=16)
        #plt.xlabel('alpha',fontsize=16)
        #plt.ylabel('cl',fontsize=16)
        #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
        plt.xlim(-0.05,1.05)
        plt.ylim(-0.20,0.20)    
        plt.axis('off')
        #plt.grid(True)
        #patch.set_facecolor('black')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.savefig('./plot/coord_%s.eps'%nname[i], format='eps')
        plt.close() 
    
        img = io.imread('./plot/coord_%s.eps'%nname[i], as_grey=True)  # load the image as grayscale
        img = util.invert(img)
        foil_mat.append(img)
        print 'image matrix size: ', img.shape      # print the size of image

info='[foil_mat,foil_fp,xx,nname,info,[x:-.05,1.05,y:-.2,.2]'    
data2=[foil_mat,foil_fp,xx,nname,info]
with open(path+'foil_param_216.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)
    
    
    
    
    

