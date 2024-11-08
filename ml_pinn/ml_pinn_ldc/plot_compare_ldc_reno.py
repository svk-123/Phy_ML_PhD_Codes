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
import  pickle
import pandas
import tensorflow as tf
from scipy import interpolate
from numpy import linalg as LA
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 





def kextract(xtmp,ytmp,u,u_pred,v,v_pred):
    
    #LinearNDinterpolator
    pD=np.asarray([xtmp,ytmp]).transpose()
        
    print ('interpolation-1...')      
    f1u=interpolate.LinearNDInterpolator(pD,u)
    xa=np.linspace(0.5,0.5,50)
    ya=np.linspace(0.01,0.99,50)
    xb=ya
    yb=xa
    u1a=np.zeros((len(ya)))
    u1b=np.zeros((len(ya)))
    for i in range(len(ya)):
        u1a[i]=f1u(xa[i],ya[i])
        u1b[i]=f1u(xb[i],yb[i])
    
    print ('interpolation-2...')      
    f2u=interpolate.LinearNDInterpolator(pD,u_pred)
    
    u2a=np.zeros((len(ya)))
    u2b=np.zeros((len(ya)))
    for i in range(len(ya)):
        u2a[i]=f2u(xa[i],ya[i])
        u2b[i]=f2u(xb[i],yb[i])
    
    print ('interpolation-3...')      
    f1v=interpolate.LinearNDInterpolator(pD,v)
    
    v1a=np.zeros((len(ya)))
    v1b=np.zeros((len(ya)))
    for i in range(len(ya)):
        v1a[i]=f1v(xb[i],yb[i])
        v1b[i]=f1v(xa[i],ya[i])
    
    print ('interpolation-4...')      
    f2v=interpolate.LinearNDInterpolator(pD,v_pred)
    
    v2a=np.zeros((len(ya)))
    v2b=np.zeros((len(ya)))
    for i in range(len(ya)):
        v2a[i]=f2v(xb[i],yb[i])
        v2b[i]=f2v(xa[i],ya[i])
    
    return(ya,u1a,u2a,xb,v1a,v2a)


def my_keras_model(val_inp):
    #load_model
    model_test=load_model('./keras_model/final_sf.hdf5') 
    kout=model_test.predict([val_inp])

    return kout

#re_train=[100,200,400,600,800,900]
#re_train=[100,200,400,600,1000,2000,4000,6000,8000,9000]    
#flist_idx=np.asarray(['100','200','300','400','500','600','700','800','900','1000','1200','1500','2000','5000','10000'])
#flist=['100','200','300','400','500','600','700','800','900','1000','1200','1500','2000']
flist=['10000']
for ii in range(len(flist)):
    
    xtmp=[]
    ytmp=[]
    p=[]
    u=[]
    v=[]
    p_pred=[]
    u_pred=[]
    v_pred=[]
    reytmp=[]
       
    #x,y,Re,u,v
    with open('./data_file_ust/cavity_Re%s.pkl'%flist[ii], 'rb') as infile:
        result = pickle.load(infile)
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    reytmp.extend(result[2])
    u.extend(result[3])
    v.extend(result[4])
    p.extend(result[5])   
        
    xtmp=np.asarray(xtmp)
    ytmp=np.asarray(ytmp)
    u=np.asarray(u)
    v=np.asarray(v)
    p=np.asarray(p) 
    reytmp=np.asarray(reytmp)/10000.  

#    val_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
#    val_out=np.concatenate((u[:,None],v[:,None],p[:,None]),axis=1)    
#    kout = my_keras_model(val_inp)
  

    #session-run
    tf.reset_default_graph
    graph = tf.get_default_graph() 
    
    #load model
    with tf.Session() as sess1:
        
        
    	path1='./tf_model/10k/case_5_pinn_100pts_8x100_ust/tf_model/'
    	new_saver1 = tf.train.import_meta_graph( path1 + 'model_0_0.000827.meta')
    	new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))
	
    	tf_dict = {'input0:0': xtmp[:,None], 'input1:0': ytmp[:,None], 'input2:0': reytmp[:,None]}
    	op_to_load1 = graph.get_tensor_by_name('prediction/BiasAdd:0')    
    	
    	#uvp-pinn
    	tout = sess1.run(op_to_load1, tf_dict)
        
    sess1.close()          
        
    #session-run
    tf.reset_default_graph
    graph = tf.get_default_graph() 
    
    #load model
    with tf.Session() as sess2:
        
        
    	path2='./tf_model/10k/case_5_nn_100pts_8x100_ust/tf_model/'
    	new_saver2 = tf.train.import_meta_graph( path2 + 'model_0_0.000130.meta')
    	new_saver2.restore(sess2, tf.train.latest_checkpoint(path2))

    	tf_dict = {'input0:0': xtmp[:,None], 'input1:0': ytmp[:,None], 'input2:0': reytmp[:,None]}
    	op_to_load2 = graph.get_tensor_by_name('prediction/BiasAdd:0')    
    
    	#uvp-nn
    	kout = sess2.run(op_to_load2, tf_dict)

    sess2.close()       
        
        
        
#idx=np.argwhere(flist==flist_idx)[0][0]
kya,ku1a,ku2a,kxb,kv1a,kv2a = kextract(xtmp,ytmp,u,kout[:,0],v,kout[:,1])
pya,pu1a,pu2a,pxb,pv1a,pv2a = kextract(xtmp,ytmp,u,tout[:,0],v,tout[:,1])

#plot
def line_plot1():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(ku1a,kya,'-og',linewidth=3,label='true')
    plt0, =plt.plot(ku2a,kya,'r',linewidth=3,label='NN')
    plt0, =plt.plot(pu2a,pya,'b',linewidth=3,label='PINN')
    plt.legend(fontsize=20)
    plt.xlabel('u-velocity',fontsize=20)
    plt.ylabel('Y',fontsize=20)
    #plt.title('%s-u'%(flist[ii]),fontsiuze=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-u'%(flist[ii]), format='png',bbox_inches='tight', dpi=100)
    plt.show() 
    
def line_plot2():
    plt.figure(figsize=(6, 5), dpi=100)
    plt0, =plt.plot(kxb,kv1a,'-og',linewidth=3,label='true')
    plt0, =plt.plot(kxb,kv2a,'r',linewidth=3,label='NN')    
    plt0, =plt.plot(pxb,pv2a,'b',linewidth=3,label='PINN')
    plt.legend(fontsize=20)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('v-velocity' ,fontsize=20)
    #plt.title('%s-v'%(flist[ii]),fontsize=16)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
    #plt.xlim(-0.1,1.2)
    #plt.ylim(-0.01,1.4)    
    plt.savefig('./plot/%s-v'%(flist[ii]), format='png',bbox_inches='tight', dpi=100)
    plt.show()     
        
    
#plot
def plot(xp,yp,zp,nc,name):

    plt.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = plt.tricontour(xp,yp,zp,nc,linewidths=0.3,colors='k',zorder=5)
    cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet,zorder=0)
    # v= np.linspace(0, 0.05, 15, endpoint=True)
    #cp = plt.tricontourf(xp,yp,zp,v,cmap=cm.jet,extend='both')
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    plt.colorbar()
    #plt.title('%s  '%flist[ii]+name)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    plt.savefig('./plot/%s'%flist[ii]+name, format='png',bbox_inches='tight', dpi=100)
    plt.show()
          
plot(xtmp,ytmp,u,20,'u-cfd')
plot(xtmp,ytmp,kout[:,0],20,'u-nn')
plot(xtmp,ytmp,tout[:,0],20,'u-pinn')
plot(xtmp,ytmp,abs(u-kout[:,0]),20,'u-error-nn')
plot(xtmp,ytmp,abs(u-tout[:,0]),20,'u-error-pinn')
#    
plot(xtmp,ytmp,v,20,'v-cfd')
plot(xtmp,ytmp,kout[:,1],20,'v-nn')
plot(xtmp,ytmp,tout[:,1],20,'v-pinn')
plot(xtmp,ytmp,abs(v-kout[:,1]),20,'v-error-nn')
plot(xtmp,ytmp,abs(v-tout[:,1]),20,'v-error-pinn')
  
    
line_plot1()
line_plot2()





