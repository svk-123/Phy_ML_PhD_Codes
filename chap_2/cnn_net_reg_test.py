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

img_rows, img_cols = 28, 28         # input image dimensions
pool_size = (2, 2)                  # size of pooling area for max pooling
prob_drop_conv = 0.2                # drop probability for dropout @ conv layer
prob_drop_hidden = 0.5              # drop probability for dropout @ fc layer

# Load MNIST dataset

f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = cPickle.load(f)
else:
    data = cPickle.load(f, encoding='bytes')
f.close()

(X_train, y_train), (X_test, y_test) = data
print('X_train original shape:', X_train.shape)

if K.image_dim_ordering() == 'th':
    # For Theano backend
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # For TensorFlow backend
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

Y_train_n=[]
for i in range(len(Y_train)):
    for j in range(10):
        if (Y_train[i,j]==1):
            Y_train_n.append(j)
Y_train_n=np.asarray(Y_train_n)

Y_test_n=[]
for i in range(len(Y_test)):
    for j in range(10):
        if (Y_test[i,j]==1):
            Y_test_n.append(j)
Y_test_n=np.asarray(Y_test_n)   

model_test=load_model('./selected_model/cnn_reg/model/final_sf.hdf5') 
out_tr=model_test.predict([X_train])[:,0] 
out_tr1=np.around(out_tr,0)

out_ts=model_test.predict([X_test])[:,0]
out_ts1=np.around(out_ts,0)


tmp=abs(Y_train_n-out_tr1)
count=0
for i in range(len(tmp)):
    if (tmp[i] > 0):
        count=count+1
print(count)
        
tmp=abs(Y_test_n-out_ts1)
count=0
for i in range(len(tmp)):
    if (tmp[i] > 0):
        count=count+1
print(count)

