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
from matplotlib import cm
import pandas
from os import listdir
from os.path import isfile, join
import cPickle as pickle

#u-t,u-rbf,v-t,v-rbf
with open('nn_layer1.pkl', 'rb') as infile:
    r1 = pickle.load(infile)
 
with open('rbf_layer1.pkl', 'rb') as infile:
    r2 = pickle.load(infile)
    

def line_plot1():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(r1[0],r1[1],'og',linewidth=2,label='true')
    plt0, =plt.plot(r1[0],r1[2],'b',linewidth=2,label='nn')
    plt0, =plt.plot(r1[0],r2[2],'r',linewidth=2,label='rbf')
    
    plt.legend(fontsize=16)
    plt.xlabel('u',fontsize=16)
    plt.ylabel('y',fontsize=16)
    plt.title('pred',fontsize=16)
    plt.savefig('pred_layer1', format='png', dpi=100)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(range(len(r1[3])),r1[3],'b',linewidth=2,label='nn-loss')
    plt0, =plt.plot(range(len(r2[3])),r2[3],'g',linewidth=2,label='rbf-loss')
    
    plt0, =plt.plot(range(len(r1[4])),r1[4],'r',linewidth=2,label='nn-val-loss')
    plt0, =plt.plot(range(len(r2[4])),r2[4],'c',linewidth=2,label='rbf-val-loss')
    
    plt.legend(fontsize=16)
    plt.xlabel('mse ',fontsize=16)
    plt.ylabel('iter' ,fontsize=16)
    plt.title('conv',fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('conv_layer1', format='png', dpi=100)
    plt.show() 



line_plot1()
line_plot2()
