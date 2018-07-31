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
       
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

#load data
xtmp=[]
ytmp=[]
reytmp=[]
aoatmp=[]
utmp=[]
vtmp=[]
ptmp=[]

flist=['Re2000']
AoA=range(7,8)
foil='n0012'

data=np.loadtxt('n0012.dat')
zz=np.zeros(len(data))
zz[:]=0.0

xu=data[0:66,0]
yu=data[0:66,1]

xl=data[66:,0]
yl=data[66:,1]

for ii in range(len(flist)):
    for jj in range(len(AoA)):
        
        #x,y,Re,u,v
        xtmp=[]
        ytmp=[]
        reytmp=[]
        aoatmp=[]
        utmp=[]
        vtmp=[]
        ptmp=[]
        
        with open('./data/%s_%s_AoA_%s.pkl'%(foil,flist[ii],AoA[jj]), 'rb') as infile:
            result = pickle.load(infile)
        xtmp.extend(result[0])
        ytmp.extend(result[1])
        reytmp.extend(result[2])
        aoatmp.extend(result[3])
        utmp.extend(result[4])
        vtmp.extend(result[5])
        ptmp.extend(result[6])   
    
        xtmp=np.asarray(xtmp)
        ytmp=np.asarray(ytmp)
        reytmp=np.asarray(reytmp)
        aoatmp=np.asarray(aoatmp)
        utmp=np.asarray(utmp)
        vtmp=np.asarray(vtmp)
        ptmp=np.asarray(ptmp) 

        # ---------ML PART:-----------#
        
        #normalize
        reytmp=reytmp/2000.
        aoatmp=aoatmp/11.
        
        val_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None],aoatmp[:,None]),axis=1)
        val_out=np.concatenate((utmp[:,None],vtmp[:,None],ptmp[:,None]),axis=1)  
        
        #load_model
        model_test=load_model('./selected_model/full/final_sf.hdf5') 
        out=model_test.predict([val_inp])    
        
        model_test1=load_model('./selected_model/local/model_sf_500_0.000156_0.000160.hdf5') 
        out=model_test1.predict([val_inp]) 
        
        #only one Re,AoA
        val_inp1=np.concatenate((xu[:,None],yu[:,None],reytmp[:len(xu),None],aoatmp[:len(xu),None]),axis=1)
        val_inp2=np.concatenate((xl[:,None],yl[:,None],reytmp[:len(xl),None],aoatmp[:len(xl),None]),axis=1)
                
        out1=model_test1.predict([val_inp1]) 
        out2=model_test1.predict([val_inp2]) 
        
        #plot
        def line_plot1():
            plt.figure(figsize=(8, 5), dpi=100)
            plt0, =plt.plot(u1a+0.0,ya,'-og',linewidth=3,label='CFD')
            plt0, =plt.plot(u1b+1.0,yb,'-og',linewidth=3)
            plt0, =plt.plot(u1c+2.0,yc,'-og',linewidth=3)
            plt0, =plt.plot(u1d+3.0,yd,'-og',linewidth=3)
            
            plt0, =plt.plot(u2a+0.0,ya,'r',linewidth=3,label='NN')
            plt0, =plt.plot(u2b+1.0,yb,'r',linewidth=3)
            plt0, =plt.plot(u2c+2.0,yc,'r',linewidth=3)
            plt0, =plt.plot(u2d+3.0,yd,'r',linewidth=3)
            
            #plt.legend(fontsize=20)
            plt.xlabel('u-velocity',fontsize=20)
            plt.ylabel('Y',fontsize=20)
            #plt.title('%s-AoA-%s-u'%(flist[ii],AoA[jj]),fontsize=16)
            plt.legend(loc='upper center',fontsize=20, bbox_to_anchor=(0.5, 1.0), ncol=2, fancybox=False, shadow=False)
            #plt.xlim(-0.1,6)
            plt.ylim(-0.01,1.3)    
            plt.savefig('./plot/%s_AoA-%s_u'%(flist[ii],AoA[jj]), format='png',bbox_inches='tight', dpi=100)
            plt.show() 
            
        def line_plot2():
            plt.figure(figsize=(8, 5), dpi=100)
            plt0, =plt.plot(v1a+0.0,ya,'-og',linewidth=3,label='CFD')
            plt0, =plt.plot(v1b+0.5,yb,'-og',linewidth=3)
            plt0, =plt.plot(v1c+1.0,yc,'-og',linewidth=3)
            plt0, =plt.plot(v1d+1.5,yd,'-og',linewidth=3)
            
            plt0, =plt.plot(v2a+0.0,ya,'r',linewidth=3,label='NN')
            plt0, =plt.plot(v2b+0.5,yb,'r',linewidth=3)
            plt0, =plt.plot(v2c+1.0,yc,'r',linewidth=3)
            plt0, =plt.plot(v2d+1.5,yd,'r',linewidth=3)  
            
            #plt.legend(fontsize=20)
            plt.xlabel('v-velocity',fontsize=20)
            plt.ylabel('Y' ,fontsize=20)
            #plt.title('%s-AoA-%s-v'%(flist[ii],AoA[jj]),fontsize=16)
            plt.legend(loc='upper center',fontsize=20,  bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
            #plt.xlim(-0.1,1.2)
            plt.ylim(-0.01,1.3)    
            plt.savefig('./plot/%s_AoA-%s_v'%(flist[ii],AoA[jj]), format='png',bbox_inches='tight', dpi=100)
            plt.show()     
            
            
        def line_plot3():
            plt.figure(figsize=(6, 5), dpi=100)
            plt0, =plt.plot(xu,pu1,'og',linewidth=3,label='CFD-upper')
            plt0, =plt.plot(xl,pl1,'oc',linewidth=3,label='CFD-lower') 
            
            '''plt0, =plt.plot(xu,pu2,'--r',linewidth=3,label='NN-upper')
            plt0, =plt.plot(xl,pl2,'--k',linewidth=3,label='NN-lower')     
            
            plt0, =plt.plot(xu,pu21,'r',linewidth=3,label='NN-upper-local')
            plt0, =plt.plot(xl,pl21,'k',linewidth=3,label='NN-lower-local') '''           
            
            plt0, =plt.plot(xu,out1[:,2],'r',linewidth=3,label='NN-upper-local')
            plt0, =plt.plot(xl,out2[:,2],'k',linewidth=3,label='NN-lower-local')              
            plt.legend(fontsize=20)
            plt.xlabel('X',fontsize=20)
            plt.ylabel('$p/P_o$' ,fontsize=20)
            #plt.title('%s-AoA-%s-p'%(flist[ii],AoA[jj]),fontsize=16)
            plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5),fontsize=20, ncol=1, fancybox=False, shadow=False)
            #plt.xlim(-0.1,1.2)
            #plt.ylim(-0.01,1.4)    
            plt.savefig('./plot/%s_AoA-%s_p'%(flist[ii],AoA[jj]), format='png',bbox_inches='tight', dpi=100)
            plt.show()     
            


        #LinearNDinterpolator
        pD=np.asarray([xtmp,ytmp]).transpose()
           
        
        #for -p
        print 'interpolation-1...'      
        f1p=interpolate.LinearNDInterpolator(pD,val_out[:,2])
       
        
        pu1=np.zeros(len(xu))
        for i in range(len(xu)):
            pu1[i]=f1p(xu[i],yu[i])
        pl1=np.zeros(len(xl))
        for i in range(len(xl)):
            pl1[i]=f1p(xl[i],yl[i])
        
        
        print 'interpolation-2...'      
        f2p=interpolate.LinearNDInterpolator(pD,out[:,2])
        
        pu2=np.zeros(len(xu))
        for i in range(len(xu)):
            pu2[i]=f2p(xu[i],yu[i])
        pl2=np.zeros(len(xl))
        for i in range(len(xl)):
            pl2[i]=f2p(xl[i],yl[i])
        
        print 'interpolation-2...'      
        f2p1=interpolate.LinearNDInterpolator(pD,out[:,2])
        
        pu21=np.zeros(len(xu))
        for i in range(len(xu)):
            pu21[i]=f2p1(xu[i],yu[i])
        pl21=np.zeros(len(xl))
        for i in range(len(xl)):
            pl21[i]=f2p1(xl[i],yl[i])
            
        #line_plot1()
        #line_plot2()
        line_plot3()





'''
plt.figure(figsize=(6, 5), dpi=200)
plt.plot(xu,yu,'k')
plt.plot(xl,yl,'k')
plt.plot([0,0],[0,1],'g',label='x=0, 0.5, 1.0, 1.5')
plt.plot([0.5,0.5],[0.05,1],'g')
plt.plot([1,1],[0,1],'g')
plt.plot([1.5,1.5],[0,1],'g')
plt.legend()
plt.xlim([-2,3])
plt.ylim([-2,2])
plt.savefig('plot.png')
plt.show()
'''

