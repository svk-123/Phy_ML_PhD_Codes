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
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from skimage import io, viewer,util 
from scipy import interpolate
import pandas as pd
from os import listdir
from os.path import isfile, join

import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)


def get_foil_mat(x,y):
    
    figure=plt.figure(figsize=(3,3))
    plt0, =plt.plot(x,y,'k',linewidth=0.5,label='true')
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.20,0.20)    
    plt.axis('off')
    #plt.grid(True)
    #patch.set_facecolor('black')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.savefig('./tmp_plot/tmp.eps', format='eps')
    plt.close() 

    img = io.imread('./tmp_plot/tmp.eps', as_grey=True)  # load the image as grayscale
    img = util.invert(img)
    return img

    
    
    
    
    

