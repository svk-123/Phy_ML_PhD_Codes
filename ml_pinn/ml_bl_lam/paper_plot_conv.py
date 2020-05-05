#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:50:55 2020

@author: vino
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas
from os import listdir
from os.path import isfile, join



#######--------------------------------------------------------------------
# Error Comparison for BCs

Re=100
path='./tf_model/case_1_re%s_dp_dv/tf_model/'%Re
suff='wos'
data1=[]
for i in range(1):
    with open(path + 'conv.dat', 'r') as infile:
        data0=infile.readlines()
    tmp=[]    
    for j in range(1,50000):
        if 'Reduce' not in data0[j]:
            tmp.append(np.asarray(data0[j].split(',')))
    tmp=np.asarray(tmp)
    tmp=tmp.astype(np.float)        
    data1.append(tmp) 
 
    
Re=100
path='./tf_model/case_1_re%s_nodp_nodv/tf_model/'%Re
data2=[]
for i in range(1):
    with open(path + 'conv.dat', 'r') as infile:
        data0=infile.readlines()
    tmp=[]    
    for j in range(1,50000):
        if 'Reduce' not in data0[j]:
            tmp.append(np.asarray(data0[j].split(',')))
    tmp=np.asarray(tmp)
    tmp=tmp.astype(np.float)        
    data2.append(tmp)     
    
    
Re=100
path='./tf_model/case_1_re%s_4s_puv/tf_model/'%Re
data3=[]
for i in range(1):
    with open(path + 'conv.dat', 'r') as infile:
        data0=infile.readlines()
    tmp=[]    
    for j in range(1,50000):
        if 'Reduce' not in data0[j]:
            tmp.append(np.asarray(data0[j].split(',')))
    tmp=np.asarray(tmp)
    tmp=tmp.astype(np.float)        
    data3.append(tmp)
    
      
l1=200
l2=250
l3=20
c=['g','b','r','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown'] 

L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):
    
    #total loss
    plt.plot(data1[i][:L,0], data1[i][:L,1] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-1')
    plt.plot(data2[i][:L,0], data2[i][:L,1] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-2')
    plt.plot(data3[i][:L,0], data3[i][:L,1] ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-3')
    
#    #MSE loss = total - gov
#    plt.plot(data1[i][:L,0], data1[i][:L,1]-data1[i][:L,3] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='1')
#    plt.plot(data2[i][:L,0], data2[i][:L,1]-data2[i][:L,3] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='2')
#    plt.plot(data3[i][:L,0], data3[i][:L,1]-data3[i][:L,3]  ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-3')

#    #gov loss
#    plt.plot(data1[i][:L,0], data1[i][:L,3] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-1')
#    plt.plot(data2[i][:L,0], data2[i][:L,3] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-2')
#    plt.plot(data3[i][:L,0], data3[i][:L,3] ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-3')
        
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('Total Loss',fontsize=20)
plt.yscale('log')
#plt.figtext(0.40, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))
#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/gov_loss_%s.png'%suff, format='png', bbox_inches='tight',dpi=300)
plt.show()
#------------------------------------------------------------------------------------------------------------------