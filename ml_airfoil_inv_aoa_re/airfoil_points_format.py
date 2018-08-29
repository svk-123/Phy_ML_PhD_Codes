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

import pandas
from os import listdir
from os.path import isfile, join

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers import merge, Input, dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)
from scipy import interpolate

#co-fomatting
path='../ml_airfoil_inv/airfoil_1600_1aoa_1re/coord_seligFmt_formatted'

indir=path
outdir='./coord_foil_1200_formatted'
outdir_itp='./foil_interp_140p'

fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()

nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])
   
'''xx=np.loadtxt('./airfoil_1600_1aoa_1re/n0012.dat')      
xxu=xx[:66,0].copy()
t1=np.linspace(0,0.001,11)
t2=np.linspace(0.002,0.05,15)
t3=np.linspace(0.055,0.1,10)
t4=np.concatenate((t1,t2,t3),axis=0)
xu_n=np.concatenate((xxu[:51],t4[::-1]),axis=0)'''

nn=70

xu_n=np.zeros((nn))
for i in range(nn):
    theta= (np.pi/float(nn))*i
    xu_n[i]=1.0-np.cos(theta)
xu_n=xu_n/xu_n.max()

xu_n=xu_n[::-1]
xl_n=xu_n[::-1]

best_name=np.genfromtxt('../ml_airfoil_inv/airfoil_1600_1aoa_1re/best_of_1343_foils.dat',dtype='str')
best_name=best_name[:1200]

# upper lower interp    
for i in range(len(fname)):
    
    if nname[i] in best_name: 
    
        print i
        '''co=np.loadtxt(path+'/%s'%fname[i],skiprows=1)
        
        coord=co.copy()
    
        coord[0,0]=1.0
        coord[-1:,0]=1.0
    
    
        ind=np.argmin(co[:,0])
        coord[ind,0]=0
        
        
        up_x=coord[:ind+1,0]
        up_y=coord[:ind+1,1]
            
        lr_x=coord[ind:,0]
        lr_y=coord[ind:,1]    
            
        # interp1
        fu = interpolate.interp1d(up_x, up_y, kind='linear')
        u_yy = fu(xu_n)
            
        fl = interpolate.interp1d(lr_x, lr_y, kind='linear')
        l_yy = fl(xl_n)'''
        
        #interp2
        '''fu = interpolate.splrep(up_x[::-1], up_y[::-1])
        u_yy = interpolate.splev(xu_n,fu)
            
        fl = interpolate.splrep(lr_x, lr_y)
        l_yy = interpolate.splev(xl_n,fl) '''
        
        
        #iterpolated
        '''fp= open(outdir_itp+"/%s"%fname[i],"w+")
        fp.write('%s\n'%nname[i])    
        
        for j in range(len(xu_n)):
            fp.write("%f %f\n"%(xu_n[j],u_yy[j])) 
            
        for j in range(1,len(xl_n)):
            fp.write("%f %f\n"%(xl_n[j],l_yy[j])) 
        fp.close()          
                
        #as it is
        fp= open(outdir+"/%s"%fname[i],"w+")
        fp.write('%s\n'%nname[i])    
        
        for j in range(len(co)):
            fp.write("%f %f\n"%(co[j,0],co[j,0])) 
        fp.close()'''  
        
        
        
        
    
        pts=np.loadtxt(outdir_itp+'/%s'%fname[i],skiprows=1)   
        
        plt.figure(figsize=(12,10))
        
        #plt.plot(co[:,0],co[:,1],'g')
        plt.plot(pts[:,0],pts[:,1],'r',ms=4)
      
        
        plt.xlabel('x',fontsize=16)
        plt.ylabel('y',fontsize=16)
        
        #plt.yscale('log')
        #plt.xticks(range(0,2001,500))
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.3,0.3])    
        
        plt.savefig('./plot/%d_%s.png'%(i,nname[i]), format='png', bbox_inches='tight',dpi=80)
        #plt.show()

    
    
    
    
    
    
    
    
##coordinate-check
#indir='./coord_seligFmt_formatted'
#fname = [f for f in listdir(indir) if isfile(join(indir, f))]
#coord=[]
#for i in range(len(fname)):
#    coord.append(np.loadtxt(indir+'/%s'%fname[i],skiprows=1))
    
    
    
    
    
    
    
    
    
    
    
    

