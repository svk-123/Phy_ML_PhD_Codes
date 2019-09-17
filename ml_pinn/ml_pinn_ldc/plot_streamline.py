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
from scipy.interpolate import griddata
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 




#re_train=[100,200,400,600,800,900]
#re_train=[100,200,400,600,1000,2000,4000,6000,8000,9000]      
flist_idx=np.asarray(['100','200','300','400','500','600','700','800','900','1000','1200','1500','2000','5000','10000'])
#flist=['100','200','300','400','500','600','700','800','900','1000','1200','1500','2000']
flist=['2000']
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
    with open('./data_file_ldc_ust/cavity_Re%s.pkl'%flist[ii], 'rb') as infile:
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
    reytmp=np.asarray(reytmp)/1000.  

    val_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
    val_out=np.concatenate((u[:,None],v[:,None],p[:,None]),axis=1)    

  
    #session-run
    tf.reset_default_graph
    graph = tf.get_default_graph() 
    
    #load model
    with tf.Session() as sess1:
        
    	path1='./tf_model/case_1_pinn_100pts_8x100_re1k_ust/tf_model_1/'
    	new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
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
        
        
    	path2='./tf_model/case_1_nn_100pts_8x100_re1k_ust/tf_model/'
    	new_saver2 = tf.train.import_meta_graph( path2 + 'model_0.meta')
    	new_saver2.restore(sess2, tf.train.latest_checkpoint(path2))

    	tf_dict = {'input0:0': xtmp[:,None], 'input1:0': ytmp[:,None], 'input2:0': reytmp[:,None]}
    	op_to_load2 = graph.get_tensor_by_name('prediction/BiasAdd:0')    
    
    	#uvp-nn
    	kout = sess2.run(op_to_load2, tf_dict)

    sess2.close()    
        


def stream_plot():
    
    fig = plt.figure(figsize=(12, 6),dpi=100)
    
    pts_x=np.linspace(0.01,0.99,200)
    pts_y=np.linspace(0.01,0.99,200)
    xx,yy=np.meshgrid(pts_x,pts_y)
    
    pts=np.concatenate((xx.flatten()[:,None],yy.flatten()[:,None]),axis=1)
    
    points=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()
    grid_y, grid_x = np.mgrid[0:1:200j, 0:1:200j]
    
    u1 = griddata(points, val_out[:,0], (grid_x, grid_y), method='linear')
    v1 = griddata(points, val_out[:,1], (grid_x, grid_y), method='linear')

    u2 = griddata(points, tout[:,0], (grid_x, grid_y), method='linear')
    v2 = griddata(points, tout[:,1], (grid_x, grid_y), method='linear')

    u3 = griddata(points, kout[:,0], (grid_x, grid_y), method='linear')
    v3 = griddata(points, kout[:,1], (grid_x, grid_y), method='linear')
    
    ax1 = fig.add_subplot(1,3,1)
    ax1.streamplot(grid_x,grid_y,u1,v1,density=4,linewidth=0.4,color='k', cmap=cm.jet,arrowsize=0.02,\
                   minlength=0.1, maxlength=4.0, zorder=0)
#    seed_points=np.array([xx.flatten(), yy.flatten()])
#    ax1.streamplot(grid_x,grid_y,u1,v1,density=4,linewidth=1,color='k', cmap=cm.jet,arrowsize=0.02,\
#                   minlength=0.1, start_points=seed_points.T, maxlength=4.0, zorder=0)    
    
        
    #ax1.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='gray',zorder=5)
    ax1.set_title('CFD')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    #plt.subplots_adjust( wspace=0.2,hspace=0.3)
    ax1.set_aspect(1)
    
    ax2 = fig.add_subplot(1,3,2)
    ax2.streamplot(grid_x,grid_y,u2,v2,density=4,linewidth=0.4,color='k', cmap=cm.jet,arrowsize=0.02,\
                   minlength=0.1, maxlength=4.0,zorder=0)
#    ax2.streamplot(grid_x,grid_y,u2,v2,density=4,linewidth=0.3,color='k', cmap=cm.jet,arrowsize=0.02,\
#                   minlength=0.1, start_points=seed_points.T, maxlength=4.0,zorder=0)
    #ax2.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='gray',zorder=5)
    ax2.set_title('PINN')
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_ylabel('Y',fontsize=16)
    ax2.set_xlim([0,1])
    ax2.set_ylim([0,1])
    ax2.set_aspect(1)
   
    
    ax3 = fig.add_subplot(1,3,3)
    ax3.streamplot(grid_x,grid_y,u3,v3,density=4,linewidth=0.4,color='k', cmap=cm.jet,arrowsize=0.02,\
                   minlength=0.1, maxlength=4.0,zorder=0)
#    ax3.streamplot(grid_x,grid_y,u2,v2,density=4,linewidth=0.3,color='k', cmap=cm.jet,arrowsize=0.02,\
#                   minlength=0.1, start_points=seed_points.T, maxlength=4.0,zorder=0)
    #ax3.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='gray',zorder=5)
    ax3.set_title('NN')
    ax3.set_xlabel('X',fontsize=16)
    ax3.set_ylabel('Y',fontsize=16)
    ax3.set_xlim([0,1])
    ax3.set_ylim([0,1])
    ax3.set_aspect(1)
    
    
      
    
    
    plt.subplots_adjust(top = 1.2, bottom = 0.1, right = 0.98, left = 0.05, hspace = 0.0, wspace = 0.25)   
    plt.savefig('./plot/stream_%s.png'%(flist[ii]),format='png',bbox_inches='tight',dpi=200)
    plt.show()
    plt.close()

stream_plot()
