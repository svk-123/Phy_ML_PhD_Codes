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
import sys

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

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

from numpy import linalg as LA
import os, shutil

"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

# ref:[data,name]

def nn_test():
    indir="./naca4/polar_val"
    fname = [f for f in listdir(indir) if isfile(join(indir, f))]
    
    #read polar
    #dataframe=pandas.read_csv(indir+'/%s'%fname[0], header=0, skiprows=None)
    #dataset = dataframe.values
    #mydata=np.asarray(dataset)
    
    name=[]   
    rey_no=[]
    data1=[]  
    for i in range(len(fname)):       
        with open(indir+'/%s'%fname[i],'r') as myfile:
            data0=myfile.readlines()
        
            if "Calculated polar for:" in data0[2]:
                    name.append(data0[2].split("NACA",1)[1]) 
        
            if "Re =" in data0[7]:
                    tmp=data0[7].split("Re =",1)[1]
                    rey_no.append(tmp.split("e",1)[0])
        
        #load alpha cl cd
        tmp_data=np.loadtxt(indir+'/%s'%fname[i],skiprows=11)
        data1.append(tmp_data[:,0:3])
    
    d1=[]
    d2=[]
    d3=[]
    #split space from name
    for i in range(len(name)):
        tmp0=name[i].split()
        tmp1=list(tmp0[0])
        d1.append(float(tmp1[0]))
        d2.append(float(tmp1[1]))
        d3.append(float(tmp1[2]+tmp1[3]))
    
    d1=np.asarray(d1)
    d2=np.asarray(d2)
    d3=np.asarray(d3)
    
    #rey_no
    for i in range(len(rey_no)):
        tmp0=rey_no[i].split()
        rey_no[i]=float(tmp0[0])
    
    for i in range(len(name)):
        tmp0=np.full(len(data1[i]),rey_no[i])
        tmp1=np.full(len(data1[i]),d1[i])
        tmp2=np.full(len(data1[i]),d2[i])
        tmp3=np.full(len(data1[i]),d3[i])
        
        data1[i]=np.concatenate((tmp0[:,None],tmp1[:,None],tmp2[:,None],tmp3[:,None],data1[i]),axis=1)
        
        
    my_error=[]    
    #load_model
    model_test=load_model('./model/final.hdf5') 
    nn_out=[]  
    for i in range(len(data1)):
        
        #Re, d1, d2, d3, alp, cl, cd
        data2=data1[i]
        
        val_inp=data2[:,0:5]
        val_inp[:,0]=val_inp[:,0]*10
        val_inp[:,3]=val_inp[:,3]/2.
        val_out=data2[:,5]
        
        
        #test-val
        out=model_test.predict([val_inp])
        nn_out.append(out)
        
        #plot
        '''
        plt.figure(figsize=(8, 5), dpi=100)
        plt0, =plt.plot(val_inp[:,4],val_out,'-og',linewidth=2,label='true')
        plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
        plt.legend(fontsize=16)
        plt.xlabel('alpha',fontsize=16)
        plt.ylabel('cl',fontsize=16)
        plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
        #plt.xlim(-0.1,1.2)
        #plt.ylim(-0.01,1.4)    
        plt.savefig('NACA%sRe=%se6'%(name[i],rey_no[i]), format='png', dpi=100)
        plt.show() 
        '''
    
        #Error
        tmp1=abs(out-val_out[:,None])
        tmp2=LA.norm(tmp1)/LA.norm(val_out)
        tmp3=(tmp2)*100
        my_error.append(tmp3)
        #print tmp3

    return (nn_out,my_error)

nn_out,nn_error=nn_test()
        
#CNN Test
path='./naca4'
with open(path+'/data_airfoil_inverse.pkl', 'rb') as infile:
    result = pickle.load(infile)
coord=result

indir=path+"/polar_val"
fname = [f for f in listdir(indir) if isfile(join(indir, f))]

name=[]   
rey_no=[]
data1=[]  
for i in range(len(fname)):       
    with open(indir+'/%s'%fname[i],'r') as myfile:
        data0=myfile.readlines()
    
        if "Calculated polar for:" in data0[2]:
                name.append(data0[2].split("NACA",1)[1]) 
    
        if "Re =" in data0[7]:
                tmp=data0[7].split("Re =",1)[1]
                rey_no.append(tmp.split("e",1)[0])
    
    #load alpha cl cd
    tmp_data=np.loadtxt(indir+'/%s'%fname[i],skiprows=11)
    data1.append(tmp_data[:,0:3])

geo_mat=True
if(geo_mat):
    for i in range(len(name)):
        tmp0=name[i].split()
        tmp1=list(tmp0[0])
               
    #rey_no
    for i in range(len(rey_no)):
        tmp0=rey_no[i].split()
        rey_no[i]=float(tmp0[0])
    tmp_name=[]
    for i in range(len(name)):
        tmp0=np.full(len(data1[i]),rey_no[i])
        tmp1=np.full(len(data1[i]),name[i])
        tmp_name.append(tmp1) # modified for prediction
        data1[i]=np.concatenate((tmp0[:,None],data1[i]),axis=1)

#name for matching with coord
# some modi done for val than trainig
new_name=[]
for i in range(len(tmp_name)):
    tmp_new_name=[]
    for j in range(len(tmp_name[i])):
        tmp_new_name.append(tmp_name[i][j].split())
    new_name.append(tmp_new_name)
  
for i in range(len(new_name)):
    for j in range(len(new_name[i])):
        new_name[i][j]='n'+'%s'%new_name[i][j][0] 

#get coord to each Re, Aoa
#strip .dat from coord
for i in range(len(coord[1])):
    coord[1][i]=coord[1][i].split('.dat',1)[0]
    
'''  
#Re, d1, d2, d3, alp, cl, cd
data2=[]
for i in range(len(name)):
    data2.extend(data1[i])
data2=np.asarray(data2)    
'''

#get coord to each Re, Aoa
co_inp=[]
veri=[]
for i in range(len(new_name)):
    tmp_co_inp=[]
    tmp_veri=[]
    for j in range(len(new_name[i])):
        for k in range(len(coord[1])):
            if (new_name[i][j]==coord[1][k]):
                tmp_co_inp.append(coord[0][k])
                tmp_veri.append(coord[1][k])
    co_inp.append(tmp_co_inp)
    veri.append(tmp_veri)              
co_inp=np.asarray(co_inp)

for i in range(len(new_name)):
    if(len(new_name[i])==len(veri[i])):
            print 'co append len OK'
    else:
        print 'co append len NOT OK'
        for j in range(len(veri[i])):
            if(new_name[i][j] != veri[i][j]):
                print 'not equal starts %d-%d'%(i,j)
                sys.exit()

'''
#input-output
val_inp1=co_inp
#Re,AoA
val_inp2=data2[:,0]
val_inp2=np.concatenate((val_inp2[:,None],data2[:,4,None]),axis=1)
val_inp2[:,0]=val_inp2[:,0]*4.0
val_inp2[:,1]=val_inp2[:,1]/10.0
val_out=data2[:,5]
'''


my_error=[]    
#load_model
model_test=load_model('./selected_model/naca4_cnn_ws/final_af_cnn.hdf5') 
  
for i in range(21):
    
    #Re, d1, d2, d3, alp, cl, cd
    data2=data1[i]
    
    val_inp1=co_inp[i]
    val_inp2=data2[:,0]
    val_inp2=np.concatenate((val_inp2[:,None],data2[:,1,None]),axis=1)
    val_inp2[:,0]=val_inp2[:,0]*4.0
    val_inp2[:,1]=val_inp2[:,1]/10.0
    val_out=data2[:,2]
    val_inp1=np.asarray(val_inp1)
    val_inp2=np.asarray(val_inp2)
    val_inp1=np.reshape(val_inp1,(len(val_inp1),216,216,1))    
    #test-val
    out=model_test.predict([val_inp1,val_inp2])
    
    
    #plot

    plt.figure(figsize=(8, 5), dpi=100)
    plt0, =plt.plot(val_inp2[:,1],val_out,'-og',linewidth=2,label='true')
    plt1, =plt.plot(val_inp2[:,1],out,'-or',linewidth=2,label='cnn')  
    plt2, =plt.plot(val_inp2[:,1],nn_out[i],'-ob',linewidth=2,label='nn') 
    plt.legend(fontsize=16)
    plt.xlabel('alpha',fontsize=16)
    plt.ylabel('cl',fontsize=16)
    plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot_out/NACA%sRe=%se6'%(name[i],rey_no[i]), format='png', dpi=100)
    plt.show() 


    #Error
    tmp1=abs(out-val_out[:,None])
    tmp2=LA.norm(tmp1)/LA.norm(val_out)
    tmp3=(tmp2)*100
    my_error.append(tmp3)
    print tmp3
nn_error=np.asarray(nn_error)
my_error=np.asarray(my_error)
print np.concatenate((my_error[:,None],nn_error[:,None]),axis=1)








