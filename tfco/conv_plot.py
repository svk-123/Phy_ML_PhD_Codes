#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

"""

import time
start_time = time.time()


# Python 3.5
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys

import os, shutil

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
#plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rc('font', family='serif')

#matplotlib.rcParams["font.family"] = "Times"
#matplotlib.rc('font',**{'family':'serif','serif':['Times']})
#matplotlib.rc('text', usetex=True)
# u'LMRoman10'


"""----------Sample--------------------"""
""" >>>with open('./datafile/to_ml/ml_allData_r0_l1.pkl', 'rb') as infile:
    >>>    result = pickle.load(infile)
    >>>Ltmp.extend(result[0])
    >>>Ttmp.extend(result[1])
    >>>bDtmp.extend(result[2])
    >>>data=[x,tb,y,coord,k,ep,rans_bij,tkedns,I]"""
"""------------------------------------"""

l1=200
l2=250
l3=20

xy=np.loadtxt('convergence1.dat')

#CNN1- fc layers
plt.figure(figsize=(6,5),dpi=100)

#plt.plot(range(len(h1l)),h1l,'r',marker='v',mfc='r',ms=12,lw=2,markevery=l1,label='2x30 Train')
plt.plot(xy[:,0],xy[:,2],'-r',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='x')
plt.plot(xy[:,0],xy[:,3],'-b',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='y')
#plt.plot(xy[:,0],xy[:,1],'-g',marker='None',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='y')


#plt.plot(range(len(h2l)),h2l,'b',marker='o',mfc='b',ms=12,lw=2,markevery=l1,label='3x30 Train')
#plt.plot(range(len(h2vl)),h2vl,'-b',marker='o',mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='3x30 Val')

#plt.plot(range(len(h3l)),h3l,'g',marker='^',mfc='g',ms=12,lw=2,markevery=l1,label='4x30 Train')
#plt.plot(range(len(h3vl)),h3vl,'-g',marker='^', mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='4x30 Val')
    
#plt.plot(range(len(h4l)),h4l,'c',marker='>',mfc='c',ms=12,lw=2,markevery=l1,label='4x50 Train')
#plt.plot(range(len(h4vl)),h4vl,'-c',marker='>', mew=1.5, mfc='None',ms=12,markevery=l2,lw=2,label='4x50 Val')

#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.25, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('Variables',fontsize=20)
#plt.yscale('log')
#plt.figtext(0.40, 0.01, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))

#plt.xlim([-50,2000])
#plt.ylim([-0.1,1.5])    
plt.savefig('lossaa.png', format='png', bbox_inches='tight',dpi=300)
plt.show()

