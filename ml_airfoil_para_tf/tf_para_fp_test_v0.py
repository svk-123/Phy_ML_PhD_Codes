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

from scipy import interpolate
from numpy import linalg as LA
import matplotlib
import tensorflow as tf

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 



# Load Data
#load data
for ii in range(1):
    with open('../ml_airfoil_para/data_file/foil_param_216_no_aug_ts.pkl', 'rb') as infile:
        result = pickle.load(infile)
    my_inp=result[0]
    my_out=result[1]
    xx=result[2]  
            
my_inp=np.asarray(my_inp)
my_out=np.asarray(my_out)
my_inp=my_inp[:100,:,:,None] 
my_out=my_out[:100]

#np.random.seed(123)
#tpara=np.zeros((10,8))
#for i in range(tpara.shape[0]):
#    for j in range(tpara.shape[1]):
#        tpara[i,j]=np.random.uniform(-0.3,0.2)
                
#session-run
graph = tf.get_default_graph() 

#load model
with tf.Session() as sess:
    
    path='./tf_models/tf_nn_para/tf_model/'
    new_saver = tf.train.import_meta_graph(path + 'model_1117_0.000002.meta' )
    new_saver.restore(sess, tf.train.latest_checkpoint(path))
           
    tf_dict = {'input:0': my_inp}
    para_to_load = graph.get_tensor_by_name('para/Tanh:0')  
    #para_to_load = graph.get_tensor_by_name('para/Tanh:0')
    tpara = sess.run(para_to_load, tf_dict)
    
    np.random.seed(123)
    ttpara=np.zeros((10,8))
    for i in range(ttpara.shape[0]):
        for j in range(ttpara.shape[1]):
            ttpara[i,j]=np.random.uniform(tpara[:,j].min(),tpara[:,j].max())          
             
    feed_dict={'para/Tanh:0': ttpara}
    op_to_load = graph.get_tensor_by_name('prediction/BiasAdd:0')   
    tout = sess.run(op_to_load, feed_dict)
    
   
#tensor_names = [t.name for op in tf.get_default_graph().get_operations() for t in op.values()]    
#fp=open('operation.txt','w')        
#for i in range(len(tensor_names)):
#    fp.write('%s \n'%tensor_names[i])
#fp.close()    
#    
#fp=open('variables.txt','w')     
#for var in tf.global_variables():
#    fp.write('%s \n'%var)   
#fp.close()
    

for k in range(10):
    print k
    yy1=tout[k][0:35]
    yy2=tout[k][35:]
    yy=np.concatenate((yy1[:,None],yy2[1:,None]))
    
    plt.figure(figsize=(6,5))
    plt.plot(xx[::-1],my_out[k][0:35],'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=1,label='True')
    plt.plot(xx,my_out[k][35:],'o',mfc='None',mew=1.5,mec='blue',ms=10,markevery=1)
    
    plt.plot(xx[::-1],tout[k][0:35],'r',lw=3,label='CNN')
    plt.plot(xx,tout[k][35:],'r',lw=3)
    
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.25,0.25])
    plt.legend(fontsize=20)
    plt.xlabel('X/c',fontsize=20)
    plt.ylabel('Y',fontsize=20)  
    #plt.axis('off')
    #plt.figtext(0.40, 0.01, '(b)', wrap=True, horizontalalignment='center', fontsize=24) 
    plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
    plt.savefig('./plot/ts_%s.png'%(k),format='png',bbox_inches='tight',dpi=100)
    plt.show()


