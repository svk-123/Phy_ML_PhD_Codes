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

from scipy import interpolate
from numpy import linalg as LA
import math       

#load data
xtmp=[]
ytmp=[]
reytmp=[]
utmp=[]
vtmp=[]

'''

flist=['Re1200']
Lc=500

#u-t,u-rbf,v-t,v-rbf
with open('./plot/cavity_mq_c200_%s.pkl'%flist[0], 'rb') as infile:
    r1 = pickle.load(infile)
 
with open('./plot/cavity_mq_c500_%s.pkl'%flist[0], 'rb') as infile:
    r2 = pickle.load(infile)
    
with open('./plot/cavity_mq_c1000_%s.pkl'%flist[0], 'rb') as infile:
    r3 = pickle.load(infile)
    




def line_plot1():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(r1[0],r1[1],'-og',linewidth=2,label='true')
    plt0, =plt.plot(r1[2],r1[3],'m',linewidth=2,label='rbf-c=200')
    plt0, =plt.plot(r2[2],r2[3],'b',linewidth=2,label='rbf-c=500')
    plt0, =plt.plot(r3[2],r3[3],'r',linewidth=2,label='rbf-c=1000')    
    
    plt.legend(fontsize=16)
    plt.xlabel('u',fontsize=16)
    plt.ylabel('y',fontsize=16)
    plt.title('%s-u'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-u_com'%(flist[0]), format='png', dpi=100)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(r1[4],r1[5],'-og',linewidth=2,label='true')
    plt0, =plt.plot(r1[6],r1[7],'m',linewidth=2,label='rbf-c=200')
    plt0, =plt.plot(r2[6],r2[7],'b',linewidth=2,label='rbf-c=500')
    plt0, =plt.plot(r3[6],r3[7],'r',linewidth=2,label='rbf-c=1000') 
    plt.legend(fontsize=16)
    plt.xlabel('x ',fontsize=16)
    plt.ylabel('v' ,fontsize=16)
    plt.title('%s-v'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-v_com'%(flist[0]), format='png', dpi=100)
    plt.show() 



line_plot1()
line_plot2()

'''

'''
flist=['Re800']
Lc=200

#u-t,u-rbf,v-t,v-rbf
with open('./plot/cavity_mq_sp1_0.01_sp2_0.5_c200_%s.pkl'%flist[0], 'rb') as infile:
    r1 = pickle.load(infile)
 
with open('./plot/cavity_mq_sp1_0.2_sp2_0.5_c200_%s.pkl'%flist[0], 'rb') as infile:
    r2 = pickle.load(infile)
    
with open('./plot/cavity_mq_sp1_0.4_sp2_0.5_c200_%s.pkl'%flist[0], 'rb') as infile:
    r3 = pickle.load(infile)
    
with open('./plot/cavity_mq_sp1_0.6_sp2_0.5_c200_%s.pkl'%flist[0], 'rb') as infile:
    r4 = pickle.load(infile)

with open('./plot/cavity_mq_sp1_0.8_sp2_0.5_c200_%s.pkl'%flist[0], 'rb') as infile:
    r5 = pickle.load(infile)

with open('./plot/cavity_mq_sp1_1.0_sp2_0.5_c200_%s.pkl'%flist[0], 'rb') as infile:
    r6 = pickle.load(infile)

def line_plot1():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(r1[0],r1[1],'-ok',linewidth=2,label='true')
    plt0, =plt.plot(r1[2],r1[3],'m',linewidth=2,label='rbf-sp=0.01')
    plt0, =plt.plot(r2[2],r2[3],'b',linewidth=2,label='rbf-sp=0.2')
    plt0, =plt.plot(r3[2],r3[3],'g',linewidth=2,label='rbf-sp=0.4')    
    plt0, =plt.plot(r4[2],r4[3],'c',linewidth=2,label='rbf-sp=0.6') 
    plt0, =plt.plot(r5[2],r5[3],'y',linewidth=2,label='rbf-sp=0.8') 
    plt0, =plt.plot(r6[2],r5[3],'r',linewidth=2,label='rbf-sp=1.0') 
    
    plt.legend(fontsize=16)
    plt.xlabel('u',fontsize=16)
    plt.ylabel('y',fontsize=16)
    plt.title('%s-u'%(flist[ii]),fontsize=16)
    plt.legend(loc='right center', bbox_to_anchor=(0.4, 1.0), ncol=1, fancybox=False, shadow=False)
    plt.xlim(-0.4,0.9)
    plt.ylim(-0.05,1.4)    
    plt.savefig('./plot/%s-u_com'%(flist[0]), format='png', dpi=100)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(r1[4],r1[5],'-ok',linewidth=2,label='true')
    plt0, =plt.plot(r1[6],r1[7],'m',linewidth=2,label='rbf-sp=0.01')
    plt0, =plt.plot(r2[6],r2[7],'b',linewidth=2,label='rbf-sp=0.2')
    plt0, =plt.plot(r3[6],r3[7],'g',linewidth=2,label='rbf-sp=0.4')    
    plt0, =plt.plot(r4[6],r4[7],'c',linewidth=2,label='rbf-sp=0.6') 
    plt0, =plt.plot(r5[6],r5[7],'y',linewidth=2,label='rbf-sp=0.8') 
    plt0, =plt.plot(r6[6],r5[7],'r',linewidth=2,label='rbf-sp=1.0') 
           
    #plt.legend(fontsize=16)
    plt.xlabel('x ',fontsize=16)
    plt.ylabel('v' ,fontsize=16)
    plt.title('%s-v'%(flist[ii]),fontsize=16)
    plt.legend(loc='left center', bbox_to_anchor=(0.5, 1.0), ncol=1, fancybox=False, shadow=False)
    plt.xlim(-0.05,1.1)
    plt.ylim(-0.6,0.6)    
    plt.savefig('./plot/%s-v_com'%(flist[0]), format='png', dpi=100)
    plt.show() 

line_plot1()
line_plot2()
'''


'''
flist=['Re800']
Lc=200

#u-t,u-rbf,v-t,v-rbf
with open('./plot/cavity_mq_sp1_0.2_sp2_0.5_c200_%s.pkl'%flist[0], 'rb') as infile:
    r1 = pickle.load(infile)
 
with open('./plot/cavity_mq_sp1_0.2_sp2_0.6_c200_%s.pkl'%flist[0], 'rb') as infile:
    r2 = pickle.load(infile)
    
with open('./plot/cavity_mq_sp1_0.2_sp2_0.8_c200_%s.pkl'%flist[0], 'rb') as infile:
    r3 = pickle.load(infile)
    
with open('./plot/cavity_mq_sp1_0.2_sp2_1.0_c200_%s.pkl'%flist[0], 'rb') as infile:
    r4 = pickle.load(infile)


def line_plot1():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(r1[0],r1[1],'-ok',linewidth=2,label='true')
    
    plt0, =plt.plot(r1[2],r1[3],'g',linewidth=2,label='rbf-sp=0.4')    
    plt0, =plt.plot(r2[2],r2[3],'c',linewidth=2,label='rbf-sp=0.6') 
    plt0, =plt.plot(r3[2],r3[3],'y',linewidth=2,label='rbf-sp=0.8') 
    plt0, =plt.plot(r4[2],r4[3],'r',linewidth=2,label='rbf-sp=1.0') 
    
    plt.legend(fontsize=16)
    plt.xlabel('u',fontsize=16)
    plt.ylabel('y',fontsize=16)
    plt.title('%s-u'%(flist[ii]),fontsize=16)
    plt.legend(loc='right center', bbox_to_anchor=(0.4, 1.0), ncol=1, fancybox=False, shadow=False)
    plt.xlim(-0.4,0.9)
    plt.ylim(-0.05,1.4)    
    plt.savefig('./plot/%s-u_com_sp2'%(flist[0]), format='png', dpi=100)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(r1[4],r1[5],'-ok',linewidth=2,label='true')
    
    plt0, =plt.plot(r1[6],r1[7],'g',linewidth=2,label='rbf-sp=0.4')    
    plt0, =plt.plot(r2[6],r2[7],'c',linewidth=2,label='rbf-sp=0.6') 
    plt0, =plt.plot(r3[6],r3[7],'y',linewidth=2,label='rbf-sp=0.8') 
    plt0, =plt.plot(r4[6],r4[7],'r',linewidth=2,label='rbf-sp=1.0') 
    
    
    
    #plt.legend(fontsize=16)
    plt.xlabel('x ',fontsize=16)
    plt.ylabel('v' ,fontsize=16)
    plt.title('%s-v'%(flist[ii]),fontsize=16)
    plt.legend(loc='left center', bbox_to_anchor=(0.5, 1.0), ncol=1, fancybox=False, shadow=False)
    plt.xlim(-0.05,1.1)
    plt.ylim(-0.6,0.6)    
    plt.savefig('./plot/%s-v_com_sp2'%(flist[0]), format='png', dpi=100)
    plt.show() 

line_plot1()
line_plot2()
'''


flist=['Re10000']
Lc=200

#u-t,u-rbf,v-t,v-rbf
with open('./plot/cavity_ga_sp1_0.2_sp2_0.4_c200_%s.pkl'%flist[0], 'rb') as infile:
    r1 = pickle.load(infile)
 
with open('./plot/cavity_ga_sp1_0.2_sp2_0.4_c500_%s.pkl'%flist[0], 'rb') as infile:
    r2 = pickle.load(infile)
    
with open('./plot/cavity_ga_sp1_0.2_sp2_0.4_c1000_%s.pkl'%flist[0], 'rb') as infile:
    r3 = pickle.load(infile)

with open('./plot/cavity_nn_%s.pkl'%flist[0], 'rb') as infile:
    r4 = pickle.load(infile)
    
def line_plot1():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(r1[0],r1[1],'-ok',linewidth=2,label='true')
    
    plt0, =plt.plot(r1[2],r1[3],'g',linewidth=2,label='rbf-c200')    
    plt0, =plt.plot(r2[2],r2[3],'b',linewidth=2,label='rbf-c500') 
    plt0, =plt.plot(r3[2],r3[3],'c',linewidth=2,label='rbf-c1000') 
    plt0, =plt.plot(r4[2],r4[3],'r',linewidth=2,label='nn') 
    
    plt.legend(fontsize=16)
    plt.xlabel('u',fontsize=16)
    plt.ylabel('y',fontsize=16)
    plt.title('%s-u'%(flist[ii]),fontsize=16)
    #plt.legend(loc='right center', bbox_to_anchor=(0.4, 1.0), ncol=1, fancybox=False, shadow=False)
    plt.legend()
    #plt.xlim(-0.4,0.9)
    #plt.ylim(-0.05,1.4)    
    plt.savefig('./plot/%s-u_com_sp2'%(flist[0]), format='png', dpi=100)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(5, 5), dpi=100)
    plt0, =plt.plot(r1[4],r1[5],'-ok',linewidth=2,label='true')
    
    plt0, =plt.plot(r1[6],r1[7],'g',linewidth=2,label='rbf-c200')    
    plt0, =plt.plot(r2[6],r2[7],'b',linewidth=2,label='rbf-c500') 
    plt0, =plt.plot(r3[6],r3[7],'c',linewidth=2,label='rbf-c1000') 
    plt0, =plt.plot(r4[6],r4[7],'r',linewidth=2,label='nn') 
 
    
    
    
    #plt.legend(fontsize=16)
    plt.xlabel('x ',fontsize=16)
    plt.ylabel('v' ,fontsize=16)
    plt.title('%s-v'%(flist[ii]),fontsize=16)
    #plt.legend(loc='left center', bbox_to_anchor=(0.5, 1.0), ncol=1, fancybox=False, shadow=False)
    plt.legend()
    #plt.xlim(-0.05,1.1)
    #plt.ylim(-0.6,0.6)    
    plt.savefig('./plot/%s-v_com_sp2'%(flist[0]), format='png', dpi=100)
    plt.show() 

line_plot1()
line_plot2()



    
    
