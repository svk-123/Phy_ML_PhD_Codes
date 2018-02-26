'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

# keras import stuff
import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K
from numpy.linalg import inv


# numpy and matplot lib
import numpy as np
import matplotlib.pyplot as plt

# training params
batch_size = 32
epochs = 100 # number of times through training set

dataset_mat=[]
dataset_inv=[]
for i in range(100):
    a=numpy.random.random((10, 10))
    b=inv(a)
    dataset_mat.append(a)
    dataset_inv.append(b)

# get train and test split
train_mat = dataset_mat[0:80]
train_inv = dataset_inv[0:80]
test_mat  = dataset_mat[80:100]
test_inv  = dataset_inv[80:100]

train_mat = np.asarray(train_mat)
train_inv = np.asarray(train_inv)
test_mat = np.asarray(test_mat)
test_inv = np.asarray(test_inv)

train_mat = train_mat.reshape(80,10,10,1)
train_inv = train_inv.reshape(80,10,10,1)
test_mat = test_mat.reshape(20,10,10,1)
test_inv = test_inv.reshape(20,10,10,1)

# construct model
inputs = Input(train_mat.shape[1:])

# 2 3x3 convolutions followed by a max pool
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

# 2 3x3 convolutions followed by a max pool
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

# 1 3x3 transpose convolution and concate conv4 on the depth dim
# ZeroPadding2D(top_pad, bottom_pad), (left_pad, right_pad)
up6 = concatenate([ZeroPadding2D(((1,0),(1,0)))(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2)), conv1], axis=3)

# 2 3x3 convolutions
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

# 1 3x3 transpose convolution and concate conv3 on the depth dim
up7 = concatenate([ZeroPadding2D(((1,0),(1,0)))(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)), conv3], axis=3)

# 2 3x3 convolutions
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

# 1 3x3 transpose convolution and concate conv3 on the depth dim
up8 = concatenate([ZeroPadding2D(((0,0),(1,0)))(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)), conv2], axis=3)

# 2 3x3 convolutions
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

# 1 3x3 transpose convolution and concate conv3 on the depth dim
up9 = concatenate([ZeroPadding2D(((0,0),(1,0)))(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)), conv1], axis=3)

# 2 3x3 convolutions
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

# final 1x1 convolutions to get to the correct depth dim (3 for 2 xy vel and 1 for pressure)
conv10 = Conv2D(1, (1, 1), activation='linear')(conv9)

# construct model
model = Model(inputs=[inputs], outputs=[conv10])

# compile the model with loss and optimizer
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['MSE'])

# train model
model.fit(train_mat, train_inv,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_mat, test_inv))

# evaluate on test set
score = model.evaluate(test_mat, test_inv, verbose=0)
print('Average Mean Squared Error:', score[0])




