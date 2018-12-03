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

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 


"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

# ref:[data,name]
path='./naca456'
data_file='/data_airfoil_inverse.pkl'

with open(path+data_file,'rb') as infile:
    result = pickle.load(infile)
coord=result

indir=path+"/polar_train"
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
#model_test=load_model('./model_cnn/final_af_cnn.hdf5') 
model_test=load_model('./selected_model/model_af_cnn_1000_0.003_0.003.hdf5')  


#spread plot
#plt.figure(figsize=(6, 5), dpi=100) 
#plt0, =plt.plot([-0.5,1.6],[-0.5,1.6],'k',lw=3) 



for i in range(71,72):
    
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
    out=np.asarray(out) 
    out=out[:,0]
    #plot
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(val_inp2[:,1]*10,val_out,'o',mfc='None',mew=1.5,mec='blue',ms=10,lw=3,label='true')
    
    

    
    plt1, =plt.plot(val_inp2[:,1]*10,out,'-or',linewidth=3,label='CNN')  
    plt.legend(fontsize=20)
    plt.xticks([0,2,4,6,8,10,12])
    plt.xlabel('AoA',fontsize=20)
    plt.ylabel('$C_l$',fontsize=20)
    #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    plt.xlim(-0.2,12.2)
    #plt.ylim(-0.01,1.4)   
    plt.savefig('./train/%s_NACA%sRe=%se6'%(i,name[i],rey_no[i]), format='png', bbox_inches='tight',dpi=100)
    plt.show()
    
    
    #spread
    #plt0, =plt.plot(val_out,out,'og') 
  
#plt.legend(fontsize=20)
'''plt.xlabel('True $C_l$',fontsize=20)
plt.ylabel('Predicted $C_l$',fontsize=20)
#plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.01,1.4) 

plt.savefig('val1', format='png',bbox_inches='tight', dpi=100)    
plt.show()'''


'''path='./selected_model/naca456_cnn_latest'
data_file='/hist.pkl'
with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
history=result[0]
#hist
plt.figure(figsize=(6,5),dpi=100)
plt.plot(range(len(history['loss'])),history['loss'],'r',lw=3,label='training error')
plt.plot(range(len(history['val_loss'])),history['val_loss'],'b',lw=3,label='validation error')
plt.legend(fontsize=20)
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
#plt.xlim([-10,5000])
#plt.ylim([-0.2,0.2])    
plt.savefig('convergence.png')
plt.show()  '''  
    
    

    
print (start_time-time.time())    
    
    
    
    
    

'''#Error
    tmp1=abs(out-val_out[:,None])
    tmp2=LA.norm(tmp1)/LA.norm(val_out)
    tmp3=(tmp2)*100
    my_error.append(tmp3)
    print tmp3'''










