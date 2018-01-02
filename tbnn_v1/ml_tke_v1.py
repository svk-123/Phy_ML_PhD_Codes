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
#


import os, shutil
folder = './model_tke/'
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
    >>>bDtmp.extend(result[2])"""
"""------------------------------------"""


ktmp=[]
tkedtmp=[]
Itmp=[]
Btmp=[]

# [x,tb,y,coord,k,ep,rans_bij,tkedns,I]
with open('./datafile/to_ml/ml_duct_Re2200_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
ktmp.extend(result[4])
tkedtmp.extend(result[7])
Itmp.extend(result[8])

with open('./datafile/to_ml/ml_duct_Re2600_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
ktmp.extend(result[4])
tkedtmp.extend(result[7])
Itmp.extend(result[8])

with open('./datafile/to_ml/ml_duct_Re2900_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
ktmp.extend(result[4])
tkedtmp.extend(result[7])
Itmp.extend(result[8])

'''with open('./datafile/to_ml/ml_duct_Re3500_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
ktmp.extend(result[4])
tkedtmp.extend(result[7])
Itmp.extend(result[8])'''

# [B]
with open('./datafile/to_ml/ml_piml_duct_Re2200_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Btmp.extend(result[0])
with open('./datafile/to_ml/ml_piml_duct_Re2600_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Btmp.extend(result[0])
with open('./datafile/to_ml/ml_piml_duct_Re2900_full.pkl', 'rb') as infile:
    result = pickle.load(infile)
Btmp.extend(result[0])
#with open('./datafile/to_ml/ml_piml_duct_Re3500_full.pkl', 'rb') as infile:
#    result = pickle.load(infile)
#Btmp.extend(result[0])

ktmp=np.asarray(ktmp)
tkedtmp=np.asarray(tkedtmp)
Itmp=np.asarray(Itmp)
Btmp=np.asarray(Btmp)

#Itmp=np.abs(Itmp)

'''
scale=[1,10000,1,10000,0.1,1]
for i in range(6):
    Itmp[:,i]=Itmp[:,i]*scale[i]
'''


# length
l=len(ktmp)


# ---------ML PART:-----------#
#shuffle data
N= len(ktmp)
I = np.arange(N)
np.random.shuffle(I)
n=9000

## Training sets
xtr1 = Btmp[I][:n]
ttr1 = tkedtmp[I][:n]


aa=Input(shape=(47,))
xx =Dense(30, kernel_initializer='random_normal')(aa)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(30)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(30)(xx)
xx=LeakyReLU(alpha=.1)(xx)
xx =Dense(30)(xx)
xx=LeakyReLU(alpha=.1)(xx)
g =Dense(1, activation='linear')(xx)

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[g])

#model = Model(inputs=[aa,t5], outputs=[y5])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, mode='min',verbose=1 ,patience=200, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=200, verbose=1, mode='auto')

filepath="./model_tke/model_tke_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=200)

# Compile model
opt = Adam(lr=2.5e-6,decay=1.0e-12)

model.compile(loss= 'mean_squared_error',optimizer= opt)


hist = model.fit(xtr1,ttr1, validation_split=0.2,epochs=3000, batch_size=100,callbacks=[reduce_lr,e_stop,chkpt],\
                 verbose=1,shuffle=False)

#save model
model.save('./model_tke/final_tke.hdf5') 


print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))


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


def pred():
    
    kp=[]
    tkedp=[]
    Ip=[]
    xyz=[]
    Bp=[]
    # [x,tb,y,coord,k,ep,rans_bij,tkedns,I]
    with open('./datafile/to_ml/ml_duct_Re3500_full.pkl', 'rb') as infile:
        result = pickle.load(infile)
    kp.extend(result[4])
    tkedp.extend(result[7])
    Ip.extend(result[8])
    xyz.extend(result[3])
    
    kp=np.asarray(kp)
    tkedp=np.asarray(tkedp)
    Ip=np.asarray(Ip)
    xyz=np.asarray(xyz)
    
    with open('./datafile/to_ml/ml_piml_duct_Re3500_full.pkl', 'rb') as infile:
        result = pickle.load(infile)   
    Bp.extend(result[0])
    Bp=np.asarray(Bp)
    
    '''
    scale=[1,10000,1,10000,0.1,1]
    for i in range(6):
        Itmp[:,i]=Itmp[:,i]*scale[i]
    '''
    
    model_test = load_model('./model_tke/final_tke.hdf5') 
    out=model_test.predict(Bp)

    out=np.asarray(out)

    plot(xyz[:,2],xyz[:,1],out[:,0],20,'pred')
    plot(xyz[:,2],xyz[:,1],tkedp,20,'dns')
    plot(xyz[:,2],xyz[:,1],kp,20,'rans')
    
    return (xyz,out)
xyz,out=pred()





