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


'''
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
    
      
l1=4000
l2=3000
l3=5000
c=['g','b','r','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown'] 

L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):
#    
#    #total loss
#    plt.plot(data1[i][:L,0], data1[i][:L,1] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-1')
#    plt.plot(data2[i][:L,0], data2[i][:L,1] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-2')
#    plt.plot(data3[i][:L,0], data3[i][:L,1] ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-3')
    
    #MSE loss = total - gov
    plt.plot(data1[i][:L,0], data1[i][:L,1]-data1[i][:L,4] ,'%s'%c[0],marker='v',mfc='g',ms=12,lw=2,markevery=l1,label='BC-1')
    plt.plot(data2[i][:L,0], data2[i][:L,1]-data2[i][:L,4] ,'%s'%c[1],marker='o',mfc='b',ms=12,lw=2,markevery=l2,label='BC-2')
    plt.plot(data3[i][:L,0], data3[i][:L,1]-data3[i][:L,4]  ,'%s'%c[2],marker='^',mfc='r',ms=12,lw=2,markevery=l3,label='BC-3')

#    #gov loss
#    plt.plot(data1[i][:L,0], data1[i][:L,4] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-1')
#    plt.plot(data2[i][:L,0], data2[i][:L,4] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-2')
#    plt.plot(data3[i][:L,0], data3[i][:L,4] ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-3')
#        
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE Loss',fontsize=20)
plt.yscale('log')
plt.figtext(0.45, 0.03, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))
#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/mse_loss_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()

L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):

    #gov loss
    plt.plot(data1[i][:L,0], data1[i][:L,4] ,'%s'%c[0],marker='v',mew=1.5, mfc='None',ms=12,lw=2,markevery=l1,label='BC-1')
    plt.plot(data2[i][:L,0], data2[i][:L,4] ,'%s'%c[1],marker='o',mew=1.5, mfc='None',ms=12,lw=2,markevery=l2,label='BC-2')
    plt.plot(data3[i][:L,0], data3[i][:L,4] ,'%s'%c[2],marker='^',mew=1.5,  mfc='None',ms=12,lw=2,markevery=l3,label='BC-3')
#        
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('N-S Loss',fontsize=20)
plt.yscale('log')
plt.figtext(0.45, 0.03, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))
#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/ns_loss_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------
'''



'''
#######--------------------------------------------------------------------
# Error Comparison for samplings

Re=100
path='./tf_model/case_1_re%s_nodp_nodv_with_samling_x8_away/tf_model/'%Re
suff='ws'
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
path='./tf_model/case_1_re%s_nodp_nodv_with_samling_x8/tf_model/'%Re
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
    
    
    
      
l1=4000
l2=6000
l3=5000
c=['g','b','r','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown'] 

L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):
#    
#    #total loss
#    plt.plot(data1[i][:L,0], data1[i][:L,1] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-1')
#    plt.plot(data2[i][:L,0], data2[i][:L,1] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-2')
#    plt.plot(data3[i][:L,0], data3[i][:L,1] ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-3')
    
    #MSE loss = total - gov
    plt.plot(data1[i][:L,0], data1[i][:L,1]-data1[i][:L,4] ,'%s'%c[0],marker='v',mfc='g',ms=12,lw=2,markevery=l1,label='S-1')
    plt.plot(data2[i][:L,0], data2[i][:L,1]-data2[i][:L,4] ,'%s'%c[1],marker='o',mfc='b',ms=12,lw=2,markevery=l2,label='S-2')

#    #gov loss
#    plt.plot(data1[i][:L,0], data1[i][:L,4] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-1')
#    plt.plot(data2[i][:L,0], data2[i][:L,4] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-2')
#    plt.plot(data3[i][:L,0], data3[i][:L,4] ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-3')
#        
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE Loss',fontsize=20)
plt.yscale('log')
plt.figtext(0.45, 0.03, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))
#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/mse_loss_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()

L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):

    #gov loss
    plt.plot(data1[i][:L,0], data1[i][:L,4] ,'%s'%c[0],marker='v',mew=1.5, mfc='None',ms=12,lw=2,markevery=l1,label='S-1')
    plt.plot(data2[i][:L,0], data2[i][:L,4] ,'%s'%c[1],marker='o',mew=1.5, mfc='None',ms=12,lw=2,markevery=l2,label='S-2')

#        
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('N-S Loss',fontsize=20)
plt.yscale('log')
plt.figtext(0.45, 0.03, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))
#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/ns_loss_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------
'''



#######--------------------------------------------------------------------
# Error Comparison hyper-parameter

Re=100
path='./tf_model/hyper/case_1_Re%s_nodp_nodv_8x50/tf_model/'%Re
suff='hyper_wos'
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
path='./tf_model/hyper/case_1_Re%s_nodp_nodv_8x150/tf_model/'%Re
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
    
      
l1=4000
l2=6000
l3=5000
c=['g','b','r','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown'] 

L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):
#    
#    #total loss
#    plt.plot(data1[i][:L,0], data1[i][:L,1] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-1')
#    plt.plot(data2[i][:L,0], data2[i][:L,1] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-2')
#    plt.plot(data3[i][:L,0], data3[i][:L,1] ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-3')
    
    #MSE loss = total - gov
    plt.plot(data1[i][:L,0], data1[i][:L,1]-data1[i][:L,4] ,'%s'%c[0],marker='v',mfc='g',ms=12,lw=2,markevery=l1,label='8 x 50')
    plt.plot(data2[i][:L,0], data2[i][:L,1]-data2[i][:L,4] ,'%s'%c[1],marker='o',mfc='b',ms=12,lw=2,markevery=l2,label='8 x 100')
    plt.plot(data3[i][:L,0], data3[i][:L,1]-data3[i][:L,4]  ,'%s'%c[2],marker='^',mfc='r',ms=12,lw=2,markevery=l3,label='8 x 120')

#    #gov loss
#    plt.plot(data1[i][:L,0], data1[i][:L,4] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='8x50')
#    plt.plot(data2[i][:L,0], data2[i][:L,4] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='8x100')
#    plt.plot(data3[i][:L,0], data3[i][:L,4] ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='8x120')
        
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE Loss',fontsize=20)
plt.yscale('log')
plt.figtext(0.45, 0.03, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))
#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/mse_loss_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()

L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):

    #gov loss
    plt.plot(data1[i][:L,0], data1[i][:L,4] ,'%s'%c[0],marker='v',mew=1.5, mfc='None',ms=12,lw=2,markevery=l1,label='8 x 50')
    plt.plot(data2[i][:L,0], data2[i][:L,4] ,'%s'%c[1],marker='o',mew=1.5, mfc='None',ms=12,lw=2,markevery=l2,label='8 x 100')
    plt.plot(data3[i][:L,0], data3[i][:L,4] ,'%s'%c[2],marker='^',mew=1.5,  mfc='None',ms=12,lw=2,markevery=l3,label='8 x 120')
        
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('N-S Loss',fontsize=20)
plt.yscale('log')
plt.figtext(0.45, 0.03, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))
#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/ns_loss_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()


#------------------------------------------------------------------------------------------------------------------



'''
#######--------------------------------------------------------------------
# Re1000 -ws

Re=20000
path='./tf_model/case_1_re%s_nodp_nodv_with_samling_x8/tf_model/'%Re
suff='Re20000'
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
 
    
      
l1=4000
l2=6000
l3=5000
c=['g','b','r','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown'] 

L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):
#    
#    #total loss
#    plt.plot(data1[i][:L,0], data1[i][:L,1] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-1')
#    plt.plot(data2[i][:L,0], data2[i][:L,1] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-2')
#    plt.plot(data3[i][:L,0], data3[i][:L,1] ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='BC-3')
    
    #MSE loss = total - gov
    plt.plot(data1[i][:L,0], data1[i][:L,1]-data1[i][:L,4] ,'%s'%c[0],marker='v',mfc='g',ms=12,lw=2,markevery=l1,label='MSE')
    plt.plot(data1[i][:L,0], data1[i][:L,4] ,'%s'%c[1],marker='v',mew=1.5, mfc='None',ms=12,lw=2,markevery=l2,label='N-S')
#    #gov loss
#    plt.plot(data1[i][:L,0], data1[i][:L,4] ,'%s'%c[0],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='8x50')
#    plt.plot(data2[i][:L,0], data2[i][:L,4] ,'%s'%c[1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='8x100')
#    plt.plot(data3[i][:L,0], data3[i][:L,4] ,'%s'%c[2],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='8x120')
        
plt.legend(loc="upper left", bbox_to_anchor=[0.5, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.yscale('log')
#plt.figtext(0.45, 0.03, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))
#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/loss_%s.tiff'%suff, format='png', bbox_inches='tight',dpi=300)
plt.show()

'''









