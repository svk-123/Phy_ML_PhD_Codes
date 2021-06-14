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



from os import listdir
from os.path import isfile, join, isdir

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

Re=1000
path='./tf_model/case_1_re%s_nodp_nodv_with_samling_x8/tf_model/'%Re

suff='Re%s_4s_puv'%Re

data=[]
for i in range(1):
    with open(path + 'conv.dat', 'r') as infile:
        data0=infile.readlines()
    tmp=[]    
    for j in range(1,50000):
        if 'Reduce' not in data0[j]:
            tmp.append(np.asarray(data0[j].split(',')))
    tmp=np.asarray(tmp)
    tmp=tmp.astype(np.float)        
    data.append(tmp)
    
xy=np.loadtxt(path + 'conv_lbfgsb.dat')
xx=range(50000,len(xy)+50000)






l1=200
l2=250
l3=20
c=['g','b','y','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown'] 

L=50000
plt.figure(figsize=(6,5),dpi=100)
for i in range(1):
    
    #plt.plot(data[i][:L,0], data[i][:L,1] - (data[i][:L,2]+data[i][:L,3]+data[i][:L,4]) ,'%s'%c[i+1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='MSE BC')
    plt.plot(data[i][:L,0], data[i][:L,2] + data[i][:L,3],'%s'%c[i],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='MSE-Adam')
    plt.plot(data[i][:L,0], data[i][:L,4] ,'%s'%c[i+1],marker='None',mfc='r',ms=12,lw=2,markevery=l1,label='Gov. Res-Adam')
    plt.plot(xx,xy[:,2],'-r',lw=2,label='MSE-L-BFGS-B')
    plt.plot(xx,xy[:,4],'-c',lw=2,label='Gov. Res-L-BFGS-B')

plt.legend(loc="upper left", bbox_to_anchor=[1, 1], ncol=1, fontsize=18, frameon=False, shadow=False, fancybox=False,title='')
plt.xlabel('Training Epochs',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.yscale('log')
#plt.figtext(0.40, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xticks(range(0,2001,500))
#plt.xlim([-50,6000])
#plt.ylim([5e-6,1e-3])    
plt.savefig('./plot/loss_%s.png'%suff, format='png', bbox_inches='tight',dpi=300)
plt.show()
