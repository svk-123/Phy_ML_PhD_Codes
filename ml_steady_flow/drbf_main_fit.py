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
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import random
import cPickle as pickle
import os, shutil



#load data
xtmp=[]
ytmp=[]
reytmp=[]
utmp=[]
vtmp=[]
ptmp=[]
flist=['Re1000','Re2000','Re3000','Re4000','Re5000','Re7000','Re8000','Re9000']
for ii in range(len(flist)):
    #x,y,Re,u,v
    with open('./data/cavity_%s.pkl'%flist[ii], 'rb') as infile:
        result = pickle.load(infile)
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    reytmp.extend(result[2])
    utmp.extend(result[3])
    vtmp.extend(result[4])
    ptmp.extend(result[5])   
    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
reytmp=np.asarray(reytmp)
utmp=np.asarray(utmp)
vtmp=np.asarray(vtmp)
ptmp=np.asarray(ptmp) 

# ---------ML PART:-----------#
#shuffle data
N= len(utmp)
I = np.arange(N)
np.random.shuffle(I)
n=70000

#normalize
reytmp=reytmp/10000.

my_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
my_out=np.concatenate((utmp[:,None],vtmp[:,None],ptmp[:,None]),axis=1)

x=my_inp.copy()
y=my_out[:,0].copy()


#kmeans
k=100
d=3
sp1=0.2

def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]

def ClusterIndicesComp(clustNum, labels_array): #list comprehension
    return np.array([i for i, x in enumerate(labels_array) if x == clustNum])

kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
c1=kmeans.cluster_centers_
c=c1.copy()

from mrbf_layers import layer_1
l1u=layer_1(x,y,c,x.shape[0],c.shape[0],3,sp1)
l1u.f_ga()

x=l1u.P
k=200
'''
kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
c2=kmeans.cluster_centers_
c=c2.copy()

l1u=layer_1(x,y,c,x.shape[0],c.shape[0],100,sp1)
l1u.f_ga()

l1u.ls_solve()
l1u.pred_f_ga()
predu=l1u.pred'''




