#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

with new parameters tanh_16_v1:

with new flow prediction network using v1.

@author: vinoth
"""
#based on parameters 


from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras
# Python 3.5
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys
import random
import pickle


from numpy import linalg as LA
import os, shutil
from scipy.optimize import minimize

import time
start_time = time.time()

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

#load data
inp_reno=[]
inp_aoa=[]
inp_para=[]

out_cm=[]
out_cd=[]
out_cl=[]
name=[]

for ii in [1]:
    
    data_file='./data_file/naca4_clcd_turb_st_3para.pkl'
    with open(data_file, 'rb') as infile:
        result = pickle.load(infile,encoding='bytes')
    out_cm.extend(result[0])   
    out_cd.extend(result[1])
    out_cl.extend(result[2])
    
    inp_reno.extend(result[3])
    inp_aoa.extend(result[4])
    inp_para.extend(result[5])
    name.extend(result[6])       
    
inp_reno=np.asarray(inp_reno)    
inp_aoa=np.asarray(inp_aoa) 
inp_para=np.asarray(inp_para)
name=np.asarray(name) 

myinp_=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
scaler=np.asarray([100000.,14.,6.,6.,30.])


#load cl-cl MLP network 
model_mlp=tf.keras.models.load_model('./turb_naca4_3para_st_6x80_tf24/model/final')


def func(para):
    
    mypara=para
    my_inp=np.reshape(mypara,(1,5))
    
    #cd, cl
    out=model_mlp.predict([my_inp])
    out=out*np.asarray([0.25,0.9])
    pred_cl=out[0,1]
    pred_cd=out[0,0]
    
    loss=(pred_cd/pred_cl)
    fp_conv.write('%f %f %f %f\n' %(loss,pred_cl,pred_cd,1/loss))

    return loss    
    

np.random.rand(123)
for j in range(10):    
    
    #starting point
    fn=np.random.randint(19000)
    #fn=100
    fp_conv=open('./plot/conv_ng_%s.dat'%fn,'w+')
    fp_conv.write('loss cl cd 1/loss\n')
      
    my_inp_=myinp_/scaler
    
    #naca4510
    p1=my_inp_[fn,0:5]
    print(p1)
    a1=my_inp_[:,0].min()
    a2=my_inp_[:,1].min()
    a3=my_inp_[:,2].min()
    a4=my_inp_[:,3].min()
    a5=my_inp_[:,4].min()
    
    
    b1=my_inp_[:,0].max()
    b2=my_inp_[:,1].max()
    b3=my_inp_[:,2].max()
    b4=my_inp_[:,3].max()
    b5=my_inp_[:,4].max()
    
    #bypass the Re,aoa values
    a1=50000/100000.
    a2=6./14.
    
    mylimit=((a1,a1),(a2,a2),(a3,b3),(a4,b4),(a5,b5))
    #mylimit=((0,6),(0,6),(6,32))
    
    res = minimize(func, x0=p1, jac=None, method = 'L-BFGS-B', bounds=mylimit, \
                   options={'disp': True, 'maxcor':100, 'ftol': 1e-16, \
                                     'eps': 1e-6, 'maxfun': 100, \
                                     'maxiter': 50, 'maxls': 100})
    
    print(name[fn])      
    print(p1*scaler[0:5])
    print(res.x*scaler[0:5])
    base=p1*scaler[0:5]
    opti=res.x*scaler[0:5]
    
    fp_res=open('./plot/res_ng_%s.dat'%fn,'w')
    fp_res.write('%f %f %f\n'%(base[2],base[3],base[4]))
    fp_res.write('%f %f %f\n'%(opti[2],opti[3],opti[4]))
    fp_res.close()
    
    
    fp_conv.close() 









'''
################# plot#############
a1=np.loadtxt('conv_1.dat',skiprows=2)
a2=np.loadtxt('conv_2.dat',skiprows=2)
a3=np.loadtxt('conv_3.dat',skiprows=2)
#a4=np.loadtxt('conv_4.dat')
#a5=np.loadtxt('conv_5.dat')

plt.figure(figsize=(6,5),dpi=100)
plt.plot(range(len(a1)),a1[:,4],'-r',label='trial-1')
plt.plot(range(len(a2))[:100],a2[:,4],'-b',label='trial-2')
plt.plot(range(len(a3)),a3[:,4],'-g',label='trial-3')
#plt.plot(range(len(a4)),a4,'-y',label='trial-4')
#plt.plot(range(len(a5))[:100],a5[:100],'-c',label='trial-5')

#plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.5),fontsize=16)
plt.legend(loc="upper left", bbox_to_anchor=[0.6, 0.5], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Iteratioins',fontsize=20)
#plt.ylabel('$C_l^{3/2}/C_d$',fontsize=20)
plt.ylabel('$C_d$',fontsize=20)
#plt.yscale('log')
#plt.figtext(0.40, 0.01, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))

#plt.xlim([-50,2000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('opti_1_cd.png', format='png', bbox_inches='tight',dpi=100)
plt.show()
'''

