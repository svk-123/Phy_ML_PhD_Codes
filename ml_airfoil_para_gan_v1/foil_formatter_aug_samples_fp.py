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
import os

import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)

path='./'

indir='./picked_cst_101/'

fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()  

nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])   

##general
coord=[]
for i in range(len(fname)):
    print ('coord',i)
    coord.append(np.loadtxt(indir+'/%s.dat'%nname[i]))
coord=np.asarray(coord)
xx=coord[0,:,0]



st=[0]
end=[len(fname)]

for iiii in range(1):
    foil_fp=[]
    foil_mat=[]
    
    folder = './plot/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            
    for i in range(st[iiii],end[iiii]):
    #for i in range(3):
        #print ('interp',i)
   

        print ('img',i)
        #plot
        figure=plt.figure(figsize=(3,3))
        plt0, =plt.plot(coord[i,:,0],coord[i,:,1],'k',linewidth=0.5,label='true')
        #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
        #plt.legend(fontsize=16)
        #plt.xlabel('alpha',fontsize=16)
        #plt.ylabel('cl',fontsize=16)
        #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
        plt.xlim(-0.05,1.05)
        plt.ylim(-0.2,0.2)    
        plt.axis('off')
        #plt.grid(True)
        #patch.set_facecolor('black')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.savefig('./plot/%s_%s_%s.eps'%(nname[i],i,iiii), format='eps')
        plt.close() 
    
        img = io.imread('./plot/%s_%s_%s.eps'%(nname[i],i,iiii), as_grey=True)  # load the image as grayscale
        img = util.invert(img)
        foil_mat.append(img)
        #print 'image matrix size: ', img.shape      # print the size of image
        foil_fp.append(coord[i,0:100,1])
    
    foil_fp=np.asarray(foil_fp)    
    info='[foil_mat,foil_fp,xx,nname,info,[x:-.05,1.05,y:-.2,.2:lw=0.5]'    
    data2=[foil_mat,foil_fp,xx,nname,info]
    with open(path+'foil_param_cst.pkl', 'wb') as outfile:
        pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)


    
    
    

