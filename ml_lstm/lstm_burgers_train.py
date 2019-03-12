import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy import linalg as la
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam
from sklearn.preprocessing import MinMaxScaler
import cPickle as pickle
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint


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


#load data
utmp=[]
ttmp=[]
x=[]

for ii in range(1):
    #x,y,Re,u,v
    with open('./data_file/burger_data.pkl', 'rb') as infile:
        result = pickle.load(infile)
    utmp.extend(result[0])
    ttmp.extend(result[1])
    x.extend(result[2])

    
utmp=np.asarray(utmp)
ttmp=np.asarray(ttmp)
x=np.asarray(x)

x_tr=utmp[0:400,:]
y_tr=utmp[1:401,:]

# ---------ML PART:-----------#

# reshape input to be 3D [samples, timesteps, features]
x_tr = np.reshape(x_tr, (x_tr.shape[0],x_tr.shape[1],1))

model = Sequential()
model.add(LSTM(1000, input_shape = (101,1)))
model.add(Dense(200,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(101))

#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, mode='min',verbose=1 ,patience=300, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=500, verbose=1, mode='auto')

filepath="./model/model_{epoch:02d}_{loss:.6f}_{val_loss:.6f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=100)

# Compile model
opt = Adam(lr=2.5e-3,decay=1.0e-12)

model.compile(loss= 'mean_squared_error',optimizer= opt)

hist = model.fit(x_tr, y_tr, validation_split=0.2,\
                 epochs=1000, batch_size=32,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=True)

#save model
model.save('./model/final_sf.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
#print("--- %s seconds ---" % (time.time() - start_time))
