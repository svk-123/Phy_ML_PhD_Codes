from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
import gzip
import cPickle 
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
import pandas
import numpy as np

import time
start_time = time.time()


batch_size = 128
nb_classes = 10
nb_epoch = 100

# Load MNIST dataset

f = gzip.open('../mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = cPickle.load(f)
else:
    data = cPickle.load(f, encoding='bytes')
f.close()

(X_train, y_train), (X_test, y_test) = data
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



# Multilayer Perceptron
# create model
aa=Input(shape=(784,))
xx =Dense(1000,  kernel_initializer='random_normal', activation='relu')(aa)
xx =Dense(1000, activation='relu')(xx)
xx =Dense(1000, activation='relu')(xx)
xx =Dense(1000, activation='relu')(xx)
g =Dense(10, activation='softmax')(xx)

#model = Model(inputs=a, outputs=g)
model = Model(inputs=[aa], outputs=[g])
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=100, min_lr=1.0e-8)

e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=200, verbose=1, mode='auto')

filepath="./model/model_sf_{epoch:02d}_{loss:.8f}_{val_loss:.8f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=100)

# Compile model
opt = Adam(lr=0.001,decay=1.0e-12)
#opt=RMSprop(lr=0.001, rho=0.9)
#model.compile(loss= 'mean_squared_error',optimizer= opt)
model.compile(loss= 'categorical_crossentropy',optimizer= opt,metrics=['accuracy'])
hist = model.fit(X_train, Y_train, validation_split=0.1,\
                 epochs=1000, batch_size=128,callbacks=[reduce_lr,e_stop,chkpt],verbose=1,shuffle=False)

#save model
model.save('./model/final_sf.hdf5') 

print"\n"
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print"\n"
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print"\n"
print("--- %s seconds ---" % (time.time() - start_time))

data1=[hist.history]
with open('./model/hist.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)



