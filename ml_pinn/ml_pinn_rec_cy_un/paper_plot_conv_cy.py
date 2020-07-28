#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 02:46:10 2020

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

Re=40
path='./tf_model/case_1_nodp_nodv_ws_ar_80_t20_20k/tf_model/'
suff='ws_x5'
data1=[]
for i in range(1):
    with open(path + 'conv.dat', 'r') as infile:
        data0=infile.readlines()
    tmp=[]    
    for j in range(1,50600):
        if 'Reduce' not in data0[j]:
            tmp.append(np.asarray(data0[j].split(',')))
    tmp=np.asarray(tmp)
    tmp=tmp.astype(np.float)        
    data1.append(tmp) 
 


#data2=[]
#for i in range(1):
#    with open(path + 'conv.dat', 'r') as infile:
#        data0=infile.readlines()
#    tmp=[]    
#    for j in range(50008,61500):
#        if 'Reduce' not in data0[j]:
#            a=np.asarray(data0[j].split(','))
#            b=np.asarray(data0[j].split(','))[3].split(' ')
#            c=np.concatenate((a[0:3],b[1:3]),axis=0)
#            tmp.append(c)
#    tmp=np.asarray(tmp)
#    tmp=tmp.astype(np.float)        
#    data2.append(tmp) 


    
      
l1=5000
l2=6000
l3=20
c=['g','b','r','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown'] 

L=100000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):
    
    #total loss
    plt.plot(data1[i][:L,0], data1[i][:L,1]-data1[i][:L,4] ,'%s'%c[0],marker='^',mfc='None',ms=11,lw=2,markevery=l1,label='MSE')
    plt.plot(data1[i][:L,0], data1[i][:L,4]                ,'%s'%c[2],marker='o',mfc='None',ms=12,lw=2,markevery=l1,label='Gov. Eq')
    
    #plt.plot(range(50008,61500), data2[i][:,1]-data2[i][:,4] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1)
    #plt.plot(range(50008,61500), data2[i][:,4]                ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1)
    #plt.plot(data1[i][:L,0], data1[i][:L,2]+data1[i][:L,3] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='MSE')

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
plt.ylabel('Loss',fontsize=20)
plt.yscale('log')
#plt.figtext(0.40, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))
#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/cy_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()
#--------------------