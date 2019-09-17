"""
@author: Maziar Raissi
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time


import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

np.random.seed(1234)
#tf.set_random_seed(1234)

class PINN(Model):
    # Initialize the class
    def __init__(self):
        
        super(PINN, self).__init__()
        
        self.d1 = Dense(2, activation='tanh')        
        self.d2 = Dense(20, activation='tanh')
        self.d3 = Dense(20, activation='tanh')
        self.d4 = Dense(20, activation='tanh')
        self.d4 = Dense(20, activation='linear')
        
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)        
        x = self.d5(x)          
        
        return x
     
        
    def myloss(self,u_tf,u_pred,f_pred):
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))
        return self.loss
             
        
            
    self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
