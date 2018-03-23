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
import pandas
from os import listdir
from os.path import isfile, join

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers import merge, Input, dot
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import cPickle as pickle
import pandas

import os, shutil

#load data
xtmp=[]
ytmp=[]
reytmp=[]
utmp=[]
vtmp=[]
#flist=['Re100','Re200','Re300','Re500','Re1000']
flist=['Re1000']
for ii in range(len(flist)):
    #x,y,Re,u,v
    with open('./data/cavity_%s.pkl'%flist[ii], 'rb') as infile:
        result = pickle.load(infile)
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    reytmp.extend(result[2])
    utmp.extend(result[3])
    vtmp.extend(result[4])
    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
reytmp=np.asarray(reytmp)
utmp=np.asarray(utmp)
vtmp=np.asarray(vtmp)


# ---------ML PART:-----------#
#shuffle data
N= len(utmp)
I = np.arange(N)
np.random.shuffle(I)
n=10000

#normalize
reytmp=reytmp/1000.

my_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
my_out=np.concatenate((utmp[:,None],vtmp[:,None]),axis=1)

## Training sets
xtr0= my_inp[I][:n]
ttr1 = my_out[I][:n]

x=xtr0[:,0:2]
y=ttr1[:,0]

#kmeans
k=200
L=len(x)
d=2
sp=0.2

#kmeans = KMeans(n_clusters=k, random_state=0).fit(x[:,0:2])
#c=kmeans.cluster_centers_
with open('./rbfout/centers.pkl', 'rb') as infile:
    result = pickle.load(infile)
print('found centers')
c=result[0]

#rbf fucntion
P=np.zeros((L,k))

#rbf = r i.e c=0
def f_dist():
    for i in range(L):
        for j in range(k):
            tmp=0
            for l in range(d):
                tmp+=(x[i,l]-c[j,l])**2
            P[i,j]=np.sqrt(tmp)

#rbf = sqrt(r^2+c^2)            
def f_mq_c():
    for i in range(L):
        for j in range(k):
            tmp=0
            for l in range(d):
                tmp+=(x[i,l]-c[j,l])**2
            P[i,j]=np.sqrt(tmp+(sp**2))
            
f_mq_c()

#ls solver-u
print('solvig LS')        
w,res,Prank,_=np.linalg.lstsq(P,y,rcond=None)

#prediction
print('predicting')  
predu=np.zeros((L))

def predu_dist():
    for i in range(L):
        tmp1=0
        for j in range(k):
            tmp2=0
            for l in range(d):
                tmp2+=(x[i,l]-c[j,l])**2
            tmp2=np.sqrt(tmp2)
            tmp1+=w[j]*tmp2
        predu[i]=tmp1

def predu_mq_c():
    for i in range(L):
        tmp1=0
        for j in range(k):
            tmp2=0
            for l in range(d):
                tmp2+=(x[i,l]-c[j,l])**2
            tmp2=np.sqrt(tmp2+(sp**2))
            tmp1+=w[j]*tmp2
        predu[i]=tmp1
        
predu_mq_c()



plt.plot(y, predu, 'o', label='rbf, c=%s'%k)
plt.plot([-0.65,1.6],[-0.65,1.6] ,'r')
plt.xlabel('true',fontsize=16)
plt.ylabel('pred',fontsize=16)
plt.legend(fontsize=16)
plt.savefig('rbf_u',format='png', dpi=100)
plt.show()

#Re, d1, d2, d3, alp, cl, cd
data_rbf=[c,w,sp]
with open('./rbfout/data_cavity_cw%s_%s_u.pkl'%(k,flist[0]), 'wb') as outfile:
    pickle.dump(data_rbf, outfile, pickle.HIGHEST_PROTOCOL)


#ls solver -v
y=ttr1[:,1]
print('solvig LS')        
w,res,Prank,_=np.linalg.lstsq(P,y,rcond=None)

#prediction
print('predicting')  
predv=np.zeros((L))

def predv_dist():
    for i in range(L):
        tmp1=0
        for j in range(k):
            tmp2=0
            for l in range(d):
                tmp2+=(x[i,l]-c[j,l])**2
            tmp2=np.sqrt(tmp2)
            tmp1+=w[j]*tmp2
        predv[i]=tmp1

def predv_mq_c():
    for i in range(L):
        tmp1=0
        for j in range(k):
            tmp2=0
            for l in range(d):
                tmp2+=(x[i,l]-c[j,l])**2
            tmp2=np.sqrt(tmp2+(sp**2))
            tmp1+=w[j]*tmp2
        predv[i]=tmp1
        
predv_mq_c()

plt.plot(y, predv, 'o', label='rbf,c=%s'%k)
plt.plot([-0.65,1.6],[-0.65,1.6] ,'r')
plt.xlabel('true',fontsize=16)
plt.ylabel('pred',fontsize=16)
plt.legend(fontsize=16)
plt.savefig('rbf_u',format='png', dpi=100)
plt.show()


#Re, d1, d2, d3, alp, cl, cd
data_rbf=[c,w,sp]
with open('./rbfout/data_cavity_cw%s_%s_v.pkl'%(k,flist[0]), 'wb') as outfile:
    pickle.dump(data_rbf, outfile, pickle.HIGHEST_PROTOCOL)
    
    
print('Prank',Prank)    
print("--- %s seconds ---" % (time.time() - start_time))
















