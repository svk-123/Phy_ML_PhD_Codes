#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

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
from keras.layers import merge, Input, dot, add, concatenate
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import cPickle as pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten,UpSampling2D
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

import os, shutil
folder = './model_cnn/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

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

with open(path + data_file, 'rb') as infile:
    result = pickle.load(infile)
coord=result

'''
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


#split space from name
geo_digit=False    
if(geo_digit):

    #need to modify some data[i] if using this
    d1=[]
    d2=[]
    d3=[]

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
    tmp_name=[]
    for i in range(len(name)):
        tmp0=np.full(len(data1[i]),rey_no[i])
        tmp1=np.full(len(data1[i]),d1[i])
        tmp2=np.full(len(data1[i]),d2[i])
        tmp3=np.full(len(data1[i]),d3[i])
        tmp4=np.full(len(data1[i]),name[i])
        tmp_name.extend(tmp4)
        data1[i]=np.concatenate((tmp0[:,None],tmp1[:,None],tmp2[:,None],tmp3[:,None],data1[i]),axis=1)
        #Re, d1, d2, d3 alp, cl, cd
        
geo_mat=True
if(geo_mat):
    #rey_no
    for i in range(len(rey_no)):
        tmp0=rey_no[i].split()
        rey_no[i]=float(tmp0[0])
    tmp_name=[]
    for i in range(len(name)):
        tmp0=np.full(len(data1[i]),rey_no[i])
        tmp1=np.full(len(data1[i]),name[i])
        tmp_name.extend(tmp1)
        data1[i]=np.concatenate((tmp0[:,None],data1[i]),axis=1)

#name for matching with coord
new_name=[]
for i in range(len(tmp_name)):
    new_name.append(tmp_name[i].split())
    
for i in range(len(new_name)):
    new_name[i]='n'+'%s'%new_name[i][0] 
    
#Re, alp, cl, cd
data2=[]
for i in range(len(name)):
    data2.extend(data1[i])
data2=np.asarray(data2)    

#get coord to each Re, Aoa
#strip .dat from coord
for i in range(len(coord[1])):
    coord[1][i]=coord[1][i].split('.dat',1)[0]
    
co_inp=[]
veri=[]
for i in range(len(new_name)):
    for j in range(len(coord[1])):
        if (new_name[i]==coord[1][j]):
            co_inp.append(coord[0][j])
            veri.append(coord[1][j])
co_inp=np.asarray(co_inp)

if (len(new_name)==len(veri)):
    print 'co append len OK'
else:
    print 'co append len NOT OK'
    for i in range(len(veri)):
        if(new_name[i] != veri[i]):
            print 'not equal starts %d'%i
            sys.exit()
#input-output
my_inp1=co_inp
#Re,AoA
my_inp2=data2[:,0]
my_inp2=np.concatenate((my_inp2[:,None],data2[:,1,None]),axis=1)
my_inp2[:,0]=my_inp2[:,0]*4
my_inp2[:,1]=my_inp2[:,1]/10.
my_out=data2[:,2]

# print dataset values
print('my_inp1 shape:', my_inp1.shape)
print('my_inp2 shape:', my_inp2.shape)
print('my_out shape:', my_out.shape)

#CNN-ML
# ---------ML PART:-----------#
#shuffle data
N= len(my_inp1)
I = np.arange(N)
np.random.shuffle(I)
n=20
'''

## Training sets
xtr1 = coord[0]
xtr1=np.reshape(xtr1,(len(xtr1),216,216,1))         
ttr1=xtr1
# print dataset values
print('xtr shape:', xtr1.shape)
print('ttr shape:', ttr1.shape)

# Multilayer Perceptron
# create model
# construct model
aa = Input([216,216,1])

# 2 3x3 convolutions followed by a max pool
conv1 = Conv2D(32, (4, 4), activation='relu', padding='same')(aa)
pool1 = MaxPooling2D(pool_size=(4, 4))(conv1)

conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(3, 3))(conv2)

conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D(pool_size=(3, 3))(conv3)

conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool3)
pool4 = MaxPooling2D(pool_size=(3, 3))(conv4)

conv5 = Conv2D(16, (2, 2), activation='relu', padding='same')(pool4)

#De-conv upsampling
#dpool5 = UpSampling2D((3,3))(conv5)

dconv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv5)
dpool4 = UpSampling2D((3, 3))(dconv4)

dconv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(dpool4)
dpool3 = UpSampling2D((3, 3))(dconv3)

dconv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(dpool3)
dpool2 = UpSampling2D((3, 3))(dconv2)

dconv1 = Conv2D(32, (4, 4), activation='relu', padding='same')(dpool2)
dpool1 = UpSampling2D((4, 4))(dconv1)

dconv0 = Conv2D(1,(4,4), activation='linear', padding='same')(dpool1)

#print
print ('aa',K.int_shape(aa))
print ('conv1',K.int_shape(conv1))
print ('pool1',K.int_shape(pool1))
print ('conv2',K.int_shape(conv2))
print ('pool2',K.int_shape(pool2))
print ('conv3',K.int_shape(conv3))
print ('pool3',K.int_shape(pool3))
print ('conv4',K.int_shape(conv4))
print ('pool4',K.int_shape(pool4))
print ('conv5',K.int_shape(conv5))
#print ('pool5',K.int_shape(pool5))

#print ('dconv5',K.int_shape(dconv5))
#print ('dpool5',K.int_shape(dpool5))
print ('dconv4',K.int_shape(dconv4))
print ('dpool4',K.int_shape(dpool4))
print ('dconv3',K.int_shape(dconv3))
print ('dpool3',K.int_shape(dpool3))
print ('dconv2',K.int_shape(dconv2))
print ('dpool2',K.int_shape(dpool2))
print ('dconv1',K.int_shape(dconv1))
print ('dpool1',K.int_shape(dpool1))
print ('dconv0',K.int_shape(dconv0))

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa],outputs=[dconv0])

#model = Model(inputs=[aa,t5], outputs=[y5])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=20, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-6, patience=30, verbose=1, mode='auto')

filepath="./model_cnn/model_enc_cnn_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"
filepath_weight="./model_cnn/weight_model_enc_cnn_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=25)

chkpt_weight= ModelCheckpoint(filepath_weight, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=True, mode='auto', period=25)
# Compile model
opt = Adam(lr=2.5e-5,decay=1e-10)

#scaler
#model.load_weights('./selected_model/naca4_cnn_ws/weight_model_af_cnn_100_0.003_0.004.hdf5')

#model.compile(loss= 'mean_squared_error',optimizer= opt)
model.compile(loss= 'binary_crossentropy',optimizer= opt)


hist = model.fit([xtr1], [ttr1], validation_split=0.2,\
                 epochs=1000, batch_size=32,callbacks=[reduce_lr,e_stop,chkpt,chkpt_weight],verbose=1,shuffle=False)

#hist = model.fit([xtr0,xtr5], [ttr5], validation_split=0.3,\
#                 epochs=10000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt,tb],verbose=1,shuffle=True)

#save model
model.save('./model_cnn/final_enc_cnn.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))











