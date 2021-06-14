from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras

import numpy as np
import pickle
import sys
import math
# from matplotlib import pyplot as plt
# #tf.enable_eager_execution()
# from mpl_toolkits import mplot3d




#load data
inp_reno=[]
inp_aoa=[]
inp_para=[]

out_cm=[]
out_cd=[]
out_cl=[]


for ii in [1]:
    
    data_file='../data_file/naca4_clcd_turb_st_3para.pkl'
    with open(data_file, 'rb') as infile:
        result = pickle.load(infile,encoding='bytes')
    out_cm.extend(result[0])   
    out_cd.extend(result[1])
    out_cl.extend(result[2])
    
    inp_reno.extend(result[3])
    inp_aoa.extend(result[4])
    inp_para.extend(result[5])
    
#    data_file='./data_file/naca4_clcd_turb_ust_3para.pkl'
#    with open(data_file, 'rb') as infile:
#        result = pickle.load(infile)
#
#	out_cm.extend(result[0])   
#	out_cd.extend(result[1])
#    	out_cl.extend(result[2])
#
#    	inp_reno.extend(result[3])
#	inp_aoa.extend(result[4])
#	inp_para.extend(result[5])
    
    

out_cm=np.asarray(out_cm)/0.20
out_cd=np.asarray(out_cd)/0.25
out_cl=np.asarray(out_cl)/0.9

inp_reno=np.asarray(inp_reno)
inp_aoa=np.asarray(inp_aoa)
inp_para=np.asarray(inp_para)/np.array([6,6,30])


# ---------ML PART:-----------#
#shuffle data
np.random.seed(123)
N= len(out_cm)
print (N)
I = np.arange(N)
np.random.shuffle(I)
n=N

#normalize
inp_reno=inp_reno/100000.
inp_aoa=inp_aoa/14.0

my_inp=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
my_out=np.concatenate((out_cd[:,None],out_cl[:,None]),axis=1)



## Training sets
xtr0= my_inp[I][:n]
ttr1 = my_out[I][:n]

xts0= my_inp[I][n:]
tts1 = my_out[I][n:]


#model
inputs = tf.keras.Input(shape=(5,))
xx = tf.keras.layers.Dense(80, activation=tf.nn.swish)(inputs)
xx = tf.keras.layers.Dense(80, activation=tf.nn.swish)(xx)
xx = tf.keras.layers.Dense(80, activation=tf.nn.swish)(xx)
xx = tf.keras.layers.Dense(80, activation=tf.nn.swish)(xx)
xx = tf.keras.layers.Dense(80, activation=tf.nn.swish)(xx)
xx = tf.keras.layers.Dense(80, activation=tf.nn.swish)(xx)
outputs = tf.keras.layers.Dense(2)(xx)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

e_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=200, verbose=1, mode='auto')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=100,\
                                                 epsilon=1.0e-8, min_lr=1.0e-8)
    
filepath="./model/model_{epoch:02d}_{loss:.6f}_{val_loss:.6f}.hdf5"

chkpt= tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=1000)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer = tf.optimizers.Adam(learning_rate=1e-4,decay=1e-12)

model.compile(optimizer=optimizer,loss='mse')
hist=model.fit(xtr0, ttr1, validation_split=0.1, batch_size=256, epochs=5000, callbacks=[reduce_lr,chkpt],verbose=1)

model.save('./model/final')
model.save('./model/final_model.hdf5')

data1=[hist.history]
with open('./model/hist.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)


