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
from matplotlib import  cm

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.layers import merge, Input, Dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import cPickle as pickle
import seaborn as sns
#
import os,sys
scriptpath = "/home/vino/miniconda2/mypy"
sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)

#load
from main_tau import L,T,bD

l=len(L)

xtr1 = T[:,:,0]
xtr2 = T[:,:,1]
xtr3 = T[:,:,2]
xtr4 = T[:,:,4]
xtr5 = T[:,:,5]
xtr6 = T[:,:,8]

bDt=np.zeros((l,6))

bDt[:,0]=bD[:,0]
bDt[:,1]=bD[:,1]
bDt[:,2]=bD[:,2]
bDt[:,3]=bD[:,4]
bDt[:,4]=bD[:,5]
bDt[:,5]=bD[:,8]



#load model
model_test = load_model('./model/final.hdf5') 
#out=model_test.predict([L,xtr1,xtr2,xtr3,xtr4,xtr5,xtr6])
out=model_test.predict([L,xtr1,xtr2,xtr3,xtr4,xtr5,xtr6])
#out=model_test.predict([L,T1m])

#  reshape
out=np.asarray(out)


def plot_results(predicted_stresses, true_stresses):
    """
    Create a plot with 9 subplots.  Each subplot shows the predicted vs the true value of that
    stress anisotropy component.  Correct predictions should lie on the y=x line (shown with
    red dash).
    :param predicted_stresses: Predicted Reynolds stress anisotropy (from TBNN predictions)
    :param true_stresses: True Reynolds stress anisotropy (from DNS)
    """
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    on_diag = [0, 3, 5]
    for i in xrange(6):
            plt.subplot(3, 3, i+1)
            ax = fig.gca()
            ax.set_aspect('equal')
            plt.plot([-1., 1.], [-1., 1.], 'r--')
            plt.scatter(true_stresses[:,i], predicted_stresses[:, i])
            plt.xlabel('True value')
            plt.ylabel('Predicted value')
            idx_1 = i / 3
            idx_2 = i % 3
            plt.title('A' + str(idx_1) + str(idx_2))
            if i in on_diag:
                plt.xlim([-1./3., 2./3.])
                plt.ylim([-1./3., 2./3.])
            else:
                plt.xlim([-0.5, 0.5])
                plt.ylim([-0.5, 0.5])
    plt.tight_layout()
    plt.show()

plot_results(out[:,:,0].transpose(),bDt)

