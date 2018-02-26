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
import matplotlib.pyplot as plt
from matplotlib import  cm

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.layers import merge, Input, Dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import cPickle as pickle
import seaborn as sns

from scipy import interpolate

import os,sys
scriptpath = "/home/vino/miniconda2/mypy"
sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)


Ltmp=[]
Ttmp=[]
bDtmp=[]
xyz=[]
k=[]
ep=[]
tkeD=[]
Btmp=[]
# for ref: data=[L,T,bD,Coord]
with open('../../tbnn_v1/datafile/to_ml/ml_duct_Re3500_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Ltmp.extend(result[0])
Ttmp.extend(result[1])
bDtmp.extend(result[2])
xyz.extend(result[3])
k.extend(result[4])
ep.extend(result[5])
tkeD.extend(result[7])
Btmp.extend(result[9])
    
bDtmp=np.asarray(bDtmp)
Ltmp=np.asarray(Ltmp)
Ttmp=np.asarray(Ttmp)
xyz=np.asarray(xyz)
k=np.asarray(k)
ep=np.asarray(ep)
tkeD=np.asarray(tkeD)
Btmp=np.asarray(Btmp)


# reduce to 6 components
l=len(Ltmp)

L=Ltmp
B=Btmp

bD=np.zeros((l,6))
bD[:,0]=bDtmp[:,0]
bD[:,1]=bDtmp[:,1]
bD[:,2]=bDtmp[:,2]
bD[:,3]=bDtmp[:,4]
bD[:,4]=bDtmp[:,5]
bD[:,5]=bDtmp[:,8]

T=np.zeros((l,10,6))
T[:,:,0]=Ttmp[:,:,0]
T[:,:,1]=Ttmp[:,:,1]
T[:,:,2]=Ttmp[:,:,2]
T[:,:,3]=Ttmp[:,:,4]
T[:,:,4]=Ttmp[:,:,5]
T[:,:,5]=Ttmp[:,:,8]

'''
#load model
model_test = load_model('../../tbnn_v1/selected_model/4b/final_duct_B.hdf5') 
outtmp=model_test.predict([B,T[:,0:4,0],T[:,0:4,1],T[:,0:4,2],T[:,0:4,3],T[:,0:4,4],T[:,0:4,5]])
    

# reshape
outtmp=np.asarray(outtmp)
outtmp=outtmp[:,:,0].transpose()

out=np.zeros((len(outtmp),9))
out[:,0]=outtmp[:,0]
out[:,1]=outtmp[:,1]
out[:,2]=outtmp[:,2]
out[:,3]=outtmp[:,1]
out[:,4]=outtmp[:,3]
out[:,5]=outtmp[:,4]
out[:,6]=outtmp[:,2]
out[:,7]=outtmp[:,4]
out[:,8]=outtmp[:,5]
'''
'''
import sys
sys.path.insert(0, '/home/vino/ml_test/ml_dns/tbnn_v1/')

from turbulencekepspreprocessor_v1 import TurbulenceKEpsDataProcessor
tdp=TurbulenceKEpsDataProcessor()

# Enforce realizability
for i in range(5):
    out = tdp.make_realizable(out)

#load model
#model_test_tke = load_model('../../tbnn_v1/model_tke/final_tke.hdf5') 
#tketmp=model_test_tke.predict(Btmp)
#tketmp=np.asarray(tketmp)
#k=tketmp[:,0]
#k=tkeD

a11=out[:,0]*2*k
a12=out[:,1]*2*k
a13=out[:,2]*2*k
a22=out[:,4]*2*k
a23=out[:,5]*2*k
a33=out[:,8]*2*k

t11=a11+(2./3.)*k
t12=a12
t13=a13
t22=a22+(2./3.)*k
t23=a23
t33=a33+(2./3.)*k

import scipy
t11=scipy.ndimage.filters.gaussian_filter(t11,0.1,mode='nearest')
t12=scipy.ndimage.filters.gaussian_filter(t12,0.1,mode='nearest')
t13=scipy.ndimage.filters.gaussian_filter(t13,0.1,mode='nearest')
t22=scipy.ndimage.filters.gaussian_filter(t22,0.1,mode='nearest')
t23=scipy.ndimage.filters.gaussian_filter(t23,0.1,mode='nearest')
t33=scipy.ndimage.filters.gaussian_filter(t33,0.1,mode='nearest')
'''
from ml_Rey_write import write_R_ml_cyclic, write_R_ml
#write_R_ml_cyclic(t11,t12,t13,t22,t23,t33)
#write_R_ml(out[:,0],out[:,1],out[:,2],out[:,4],out[:,5],out[:,8])
write_R_ml(bDtmp[:,0],bDtmp[:,1],bDtmp[:,2],bDtmp[:,4],bDtmp[:,5],bDtmp[:,8])

from ml_tke_write import write_tke_ml
write_tke_ml(k)

#plot
def plot(x,y,z,nc,name):
    fig=plt.figure(figsize=(6, 5), dpi=100)
    ax=fig.add_subplot(111)
    #cp = ax.tricontourf(x, y, z,np.linspace(-0.3,0.3,30),extend='both')
    cp = ax.tricontourf(x, y, z,30,extend='both')
    #cp.set_clim(-0.2,0.2)
    #plt.xlim([-1, 0])
    #plt.ylim([-1, 0])
     
    cbar=plt.colorbar(cp)
    plt.title(name)
    plt.xlabel('Z ')
    plt.ylabel('Y ')
    #plt.savefig(name +'.png', format='png', dpi=100)
    plt.show()



#import scipy
#out=scipy.ndimage.filters.gaussian_filter(out,0.1,mode='nearest')

'''
z=xyz[:,2]
y=xyz[:,1]

plot(z,y,t11,20,'t11')
plot(z,y,t12,20,'t12')
plot(z,y,t13,20,'t13')
plot(z,y,t22,20,'t22')
plot(z,y,t23,20,'t23')
plot(z,y,t33,20,'t33')
  
plot(z,y,k,20,'k')
'''

















