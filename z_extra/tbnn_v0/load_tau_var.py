#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 19:05:27 2017

@author: vino
"""

# imports
import os
import glob

import numpy as np
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate


#load
import cPickle as pickle
# for ref: data1 = [tauD,Tc,L]
with open('./py_tau_interpret/data1.pkl', 'rb') as infile:
    result = pickle.load(infile)
