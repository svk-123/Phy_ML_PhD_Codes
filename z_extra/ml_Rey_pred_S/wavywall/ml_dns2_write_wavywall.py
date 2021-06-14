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
import pandas as pd
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
with open('../../tbnn_v1/datafile/to_ml/ml_wavywall_Re6760_full.pkl', 'rb') as infile:
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



#load model
model_test = load_model('../../tbnn_v1/selected_model/final_cbfs_B_p09_p006.hdf5') 
outtmp=model_test.predict([B,T[:,:,0],T[:,:,1],T[:,:,2],T[:,:,3],T[:,:,4],T[:,:,5]])
   
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



import sys
sys.path.insert(0, '/home/vino/ml_test/ml_dns/tbnn_v1/')

from turbulencekepspreprocessor_v1 import TurbulenceKEpsDataProcessor
tdp=TurbulenceKEpsDataProcessor()

# Enforce realizability
for i in range(5):
    out = tdp.make_realizable(out)



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



from ml_Rey_write_wavywall import write_R_ml
write_R_ml(t11,t12,t13,t22,t23,t33,xyz[:,0],xyz[:,1],xyz[:,2])


def rr_inp():
    "rans data to rans data Rey interpolation (from slice to wholedata)"
    print 'run...get_rans_cbfs...'    
    data = np.loadtxt('hill_Re10595_sort_full.txt', skiprows=1)
    x,y,z=data[:,0],data[:,1],data[:,2]   
    rxx,rxy,rxz=data[:,19],data[:,20],data[:,21]
    ryx,ryy,ryz=data[:,22],data[:,23],data[:,24]
    rzx,rzy,rzz=data[:,25],data[:,26],data[:,27]    
    write_R_ml(rxx,rxy,rxz,ryy,ryz,rzz,x,y,z)

def dr_inp():
    "dns data to rans data Rey interpolation (from slice to wholedata)"
    
    path_d='../../dns_data/hill/Re10595/hill_train.dat'
    dataframe = pd.read_csv(path_d,sep='\s+',header=None, skiprows=20)
    dataset = dataframe.values
    data=np.asarray(dataset)
        
    """VARIABLES = 0-x,1-y,2-p,3-u/Ub,4-v/Ub,5-w/Ub,6-nu_t/nu,7-uu/Ub^2,8-vv/Ub^2,9-ww/Ub^2,10-uv/Ub^2.
                       11-uw/Ub^2,12-vw/Ub^2,13-k/Ub^2"""
                      
    xD,yD,p,u,v,w,nu,uu,vv,ww,uv,uw,vw,k = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],\
                                             data[:,6],data[:,7],data[:,8],data[:,9],data[:,10],data[:,11],data[:,12],data[:,13]
                                             
    write_R_ml(uu,uv,uw,vv,vw,ww,xD,yD,xD)                                         


#from ml_tke_write import write_tke_ml
#write_tke_ml(k)

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

x=xyz[:,0]
y=xyz[:,1]
  
plot(x,y,t11,20,'k')



print("--- %s seconds ---" % (time.time() - start_time))








