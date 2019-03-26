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
from matplotlib import cm
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

from scipy import interpolate
from numpy import linalg as LA
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
plt.rc('font', family='serif')

##load data
#inp_x=[]
#inp_y=[]
#inp_reno=[]
#inp_aoa=[]
#inp_para=[]
#
#out_p=[]
#out_u=[]
#out_v=[]
#
#
##calculate error norm
#train_l2_p=[]
#train_l2_u=[]
#train_l2_v=[]
#
#
##load data
##with open('./data_file/ph_1_test/foil_aoa_nn_p16_ph_1_ts_1.pkl', 'rb') as infile:
#with open('./data_file/foil_aoa_nn_nacan_lam_ts_2.pkl', 'rb') as infile:
#    result = pickle.load(infile)
#
#inp_x.extend(result[0])   
#inp_y.extend(result[1])
#inp_para.extend(result[2])
#inp_reno.extend(result[3])
#inp_aoa.extend(result[4])
#
#out_p.extend(result[5])
#out_u.extend(result[6])
#out_v.extend(result[7])
#
#co=result[8]
#fxy=result[9]
#name=result[9]
#
#inp_x=np.asarray(inp_x)
#inp_y=np.asarray(inp_y)
#inp_reno=np.asarray(inp_reno)
#inp_aoa=np.asarray(inp_aoa)
#inp_para=np.asarray(inp_para)
#
#out_p=np.asarray(out_p)
#out_u=np.asarray(out_u)
#out_v=np.asarray(out_v)
#
#
#
#def find_nearest(array, value):
#    array = np.asarray(array)
#    idx = (np.abs(array - value)).argmin()
#    return idx
#
#
##load_model
#start_time = time.time()
#model_test=load_model('./selected_model/case_9_naca_lam_1/model_sf_65_0.00000317_0.00000323.hdf5') 
#end_time = time.time()
#
#print len(inp_x)
#
#for i in range(len(inp_x)):
#
#    print i
#    #normalize
#    inp_reno[i]=inp_reno[i]/2000.
#    inp_aoa[i]=inp_aoa[i]/14.0
#    
#    val_inp=np.concatenate((inp_x[i][:,None],inp_y[i][:,None],inp_reno[i][:,None],inp_aoa[i][:,None],inp_para[i][:,:]),axis=1)
#    val_out=np.concatenate((out_p[i][:,None],out_u[i][:,None],out_v[i][:,None]),axis=1)
#   
#    start_time = time.time()
#    out=model_test.predict([val_inp]) 
#    end_time = time.time()     
#
#    
#    tmp=val_out[:,0]-out[:,0]
#    train_l2_p.append( (LA.norm(tmp)/LA.norm(out[:,0]))*100 )
#
#    tmp=val_out[:,1]-out[:,1]
#    train_l2_u.append( (LA.norm(tmp)/LA.norm(out[:,1]))*100 )
#
#    tmp=val_out[:,2]-out[:,2]
#    train_l2_v.append( (LA.norm(tmp)/LA.norm(out[:,2]))*100 )
#   
## ref:[x,y,z,ux,uy,uz,k,ep,nut]
#info=['l2_p, u, v']
#
#data1 = [train_l2_p,train_l2_u, train_l2_v, info ]
#
#with open('test_l2_2.pkl', 'wb') as outfile1:
#    pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)  
  

#plot -test error
p=[]
u=[]
v=[]
for i in range(1,3):
    with open('./error/test_l2_%s.pkl'%i, 'rb') as infile:
        result = pickle.load(infile)
    
    p.extend(result[0])   
    u.extend(result[1])
    v.extend(result[2])

p_avg=sum(p)/len(p)
u_avg=sum(u)/len(u)
v_avg=sum(v)/len(v)

p=np.asarray(p)
u=np.asarray(u)
v=np.asarray(v)

tot=(p+u+v)/3.0

tot_avg=sum(tot)/len(tot)    

#error plot
plt.figure(figsize=(6,5),dpi=100)
plt.hist(tot, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.ylabel('Number of Samples',fontsize=20)
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.figtext(0.40, 0.01, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xlim([-0.001,4.0])
plt.savefig('ts_tot.tiff',format='tiff', bbox_inches='tight',dpi=300)
plt.show()    





#plot -train error
p=[]
u=[]
v=[]
for i in range(1):
    with open('./error/train_l2.pkl', 'rb') as infile:
        result = pickle.load(infile)
    
    p.extend(result[0])   
    u.extend(result[1])
    v.extend(result[2])

p_avg=sum(p)/len(p)
u_avg=sum(u)/len(u)
v_avg=sum(v)/len(v)

p=np.asarray(p)
u=np.asarray(u)
v=np.asarray(v)

tot=(p+u+v)/3.0

tot_avg=sum(tot)/len(tot)    

#error plot
plt.figure(figsize=(6,5),dpi=100)
plt.hist(tot, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.ylabel('Number of Samples',fontsize=20)
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.figtext(0.40, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xlim([-0.001,4.0])
plt.savefig('tr_tot.tiff',format='tiff', bbox_inches='tight',dpi=300)
plt.show()





    
##error plot
#plt.figure(figsize=(6,5),dpi=100)
#plt.hist(u, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
#plt.ylabel('Number of Samples',fontsize=20)
#plt.xlabel('$L_2$ relative error(%)',fontsize=20)
##plt.xlim([-0.001,4.0])
#plt.savefig('ts_u.png',format='png', bbox_inches='tight',dpi=200)
#plt.show() 
#
##error plot
#plt.figure(figsize=(6,5),dpi=100)
#plt.hist(v, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
#plt.ylabel('Number of Samples',fontsize=20)
#plt.xlabel('$L_2$ relative error(%)',fontsize=20)
##plt.xlim([-0.001,4.0])
#plt.savefig('ts_v.png',format='png', bbox_inches='tight',dpi=200)
#plt.show() 




