# Generate and plot the contour of an airfoil 
# using the PARSEC parameterization

# Repository & documentation:
# http://github.com/dqsis/parsec-airfoils
# -------------------------------------


# Import libraries
from __future__ import division
from math import sqrt, tan, pi
import numpy as np


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
import pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K

from numpy import linalg as LA
import os, shutil
from skimage import io, viewer,util 
from scipy.optimize import minimize

model_para=load_model('./model_cnn_v1/case_1_p8_tanh_include_naca4_v2/model_cnn/final_cnn.hdf5') 
get_c= K.function([model_para.layers[16].input],  [model_para.layers[19].output]) 
xx=np.loadtxt('xx_101.dat')

def synthesize(p2,n_points=100):
    
    para1=p2
    para1=np.reshape(para1,(1,8))
    c1 = get_c([para1])[0][0,:]
    c1=c1*0.2
    c2=np.concatenate((c1[0:100],-c1[0:1]),axis=0)
    return (np.asarray([xx,c2]).transpose())
