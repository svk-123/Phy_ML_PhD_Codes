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
import numpy.linalg as LA
import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)


path='./airfoil_1600_1aoa_1re/'

data_file='cp_foil_1600.pkl'

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
cp_up=result[0]
cp_lr=result[1]
foil=result[2]
xx=result[3]
name=result[4]

img_mat_cp=[]
img_mat_cp_n=[]
for i in range(100,300):
    
    cp=np.concatenate((cp_up[i],cp_lr[i]),axis=0)

    figure=plt.figure(figsize=(3,3))
    plt0, =plt.plot(cp[:,0],cp[:,1],'k',linewidth=2,label='true')
    plt.xlim(-0.05,1.05)
    plt.ylim(-3.0,1.2)    
    plt.axis('off')
    #plt.grid(True)
    #patch.set_facecolor('black')
    plt.savefig('./plot_out_cp/cp_%s'%i, format='png')
    plt.show() 
               
    img_cp = io.imread('./plot_out_cp/cp_%s'%i, as_grey=True)  # load the image as grayscale
    img_mat_cp.append(img_cp)
    print 'image matrix size: ', img_cp.shape      # print the size of image
        
    bor=np.argwhere(img_cp ==0.0)
    img_cp_n=img_cp.copy()

    for m in range(216):
        for n in range(216):
            if (([m,n] == bor).all(1).any() == False):
         
                dist=LA.norm((bor-[m,n]),axis=1).min()
                
                img_cp_n[m,n]=dist*0.01

    
    img_mat_cp_n.append(img_cp_n)
    plt.imshow(img_cp_n)
    plt.savefig('./plot_out_cp/cp_d_%s'%i, format='png')
    plt.axis('off')
  

data2=[cp_up,cp_lr,img_mat_cp_n,xx,name]
with open(path+'cp_foil_1600_dist_300.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)    
    
    

