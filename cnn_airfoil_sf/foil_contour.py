#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

@author: vinoth
"""
from __future__ import print_function

import time
start_time = time.time()

# Python 3.5
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from skimage import io, viewer,util 
from scipy import interpolate
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy.linalg as LA
import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)

path='./plotc'

indir=path

fname = [f for f in listdir(indir) if isfile(join(indir, f))]

nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.png')[0])





for i in range(1):
    
    pts=np.loadtxt('./airfoil_data/foil200/naca643418.dat',skiprows=1)
               
    img_foil = io.imread(path+'/%s.png'%(nname[i]))  # load the image as grayscale
    print('image matrix size: ', img_foil.shape )     # print the size of image
    
    

    plt.imshow(img_foil[:,:,3])