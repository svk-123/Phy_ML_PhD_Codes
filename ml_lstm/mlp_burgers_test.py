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
import pickle
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.models import load_model

import os, shutil

#load data
utmp=[]
ttmp=[]
x=[]

for ii in range(1):
    #x,y,Re,u,v
    with open('./data_file/burger_data_more.pkl', 'rb') as infile:
        result = pickle.load(infile,encoding='bytes')
    utmp.extend(result[0])
    ttmp.extend(result[1])
    x.extend(result[2])

    
utmp=np.asarray(utmp)
ttmp=np.asarray(ttmp)
x=np.asarray(x)

x_tr=utmp[0:2499,:]
y_tr=utmp[1:2500,:]

# ---------ML PART:-----------#

# reshape input to be 3D [samples, timesteps, features]
x_tr_mlp = np.concatenate((x_tr[:,:],ttmp[0:2499,None]),axis=1)
#load_model
model_test_mlp=load_model('./selected_model/mlp_1/model_1000_0.000001_0.000221.hdf5') 
out_mlp=model_test_mlp.predict(x_tr_mlp) 

#lstm
x_tr_lstm = np.reshape(x_tr, (x_tr.shape[0],x_tr.shape[1],1))
model_test_lstm=load_model('./selected_model/case_1_2x500_4x500/model_800_0.000001_0.000024.hdf5') 
out_lstm=model_test_lstm.predict(x_tr_lstm) 


for k in range(2000,2200,10):

    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(x,y_tr[k,:],'ro',markersize=8,label='true')
    plt.plot(x,out_lstm[k,:],'b',lw=3,label='lstm')
    plt.plot(x,out_mlp[k,:],'g',lw=3,label='mlp')
    plt.legend(fontsize=20)
    plt.xlabel('X',fontsize=20)
    plt.ylabel('Y',fontsize=20)  
    #plt.axis('off')
    plt.tight_layout()
    plt.savefig('./plot/ts_%04d.png'%(k), bbox_inches='tight',dpi=100)
    plt.show()


