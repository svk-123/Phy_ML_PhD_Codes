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
import pandas

import os, shutil
folder = './model/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

#read excel file
dataframe=pandas.read_excel('airfoil_v1.xlsx', sheet_name=0, header=0, skiprows=None)
dataset = dataframe.values
mydata=np.asarray(dataset)

#format the data
tmpcol=mydata[:,6]
mylist=[]
for i in range(len(tmpcol)):
    mylist.append(map(float,tmpcol[i].split(',')))
mylist=np.asarray(mylist)    
mydata=np.concatenate((mydata, mylist), axis=1)
mydata=np.delete(mydata,[0,6],axis=1)

tmplist=[]
for i in range(len(mydata)):
    if(mydata[i,4]==5):
        tmplist.append(i)
        
mydata=np.delete(mydata,tmplist,axis=0)        

#train - validation split
my_inp=mydata[:-10,[0,1,2,3,6]]
my_out=mydata[:-10,5]

val_inp=mydata[-10:,[0,1,2,3,6]]
val_out=mydata[-10:,5]

#normalize
my_inp[:,2]=my_inp[:,2]/2.
val_inp[:,2]=val_inp[:,2]/2.

my_inp[:,3]=my_inp[:,3]/50000.
val_inp[:,3]=val_inp[:,3]/50000.

my_out=my_out/100.
val_out=val_out/100.

#Network

# ---------ML PART:-----------#
#shuffle data
N= len(my_inp)
I = np.arange(N)
np.random.shuffle(I)
n=124

## Training sets
xtr0= my_inp[I][:n]
ttr1 = my_out[I][:n]



# Multilayer Perceptron
# create model

aa=Input(shape=(5,))
xx =Dense(30,  kernel_initializer='random_normal', activation='tanh')(aa)
xx =Dense(30, activation='tanh')(xx)
xx =Dense(30, activation='tanh')(xx)
g =Dense(1, activation='linear')(xx)

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[g])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, mode='min',verbose=1 ,patience=1000, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=1000, verbose=1, mode='auto')

filepath="./model/model_{epoch:02d}_{loss:.3f}_{val_loss:.3f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=1000)

# Compile model
opt = Adam(lr=2.5e-5,decay=1.0e-12)

model.compile(loss= 'mean_squared_error',optimizer= opt)


hist = model.fit([xtr0], [ttr1], validation_split=0.1,\
                 epochs=25000, batch_size=16,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=False)

#save model
model.save('./model/final.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))


#test-val
model=load_model('./model/final.hdf5')
out=model.predict([val_inp])

#re-normalize
val_out=val_out*100
out=out*100

#plot
plt.figure(figsize=(8, 5), dpi=100)
plt0, =plt.plot(val_out,'og',label='true')
plt1, =plt.plot(out,'or',label='nn')  
plt.legend()
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.01,1.4)    
#plt.savefig('u_duct_ml', format='png', dpi=100)
plt.show() 

























