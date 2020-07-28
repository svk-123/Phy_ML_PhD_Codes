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

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import pickle

import pandas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from scipy import interpolate
from numpy import linalg as LA
import matplotlib

#matplotlib.rc('xtick', labelsize=18) 
#matplotlib.rc('ytick', labelsize=18) 

##load data
#xtmp=[]
#ytmp=[]
#p=[]
#u=[]
#v=[]
#p_pred=[]
#u_pred=[]
#v_pred=[]
#

#flist=['re40']
#suff='bl'
#for ii in range(len(flist)):
#    #x,y,Re,u,v
#    with open('./data_file/BL_0503.pkl', 'rb') as infile:
#        result = pickle.load(infile)
#    xtmp.extend(result[0])
#    ytmp.extend(result[1])
#    p.extend(result[2])
#    u.extend(result[3])
#    v.extend(result[4])    
#
#    

#xtmp=np.asarray(xtmp)
#ytmp=np.asarray(ytmp)
#p=np.asarray(p)
#u=np.asarray(u)
#v=np.asarray(v)

Re=100
suff='re%s_nodp_nodv_x8_50'%Re    
xy=np.loadtxt('./data_file/Re%s/bl_internal_combined.dat'%Re,skiprows=1)



nu_=1.0/float(Re)
x1=5
Rex1=x1/nu_
d5=4.91*x1/np.sqrt(Rex1)


I=[]
for i in range(len(xy)):
    if (xy[i,1] <= d5 ):
        I.append(i)



val_inp=np.concatenate((xy[I,0:1],xy[I,1:2]),axis=1)
val_out=np.concatenate((xy[I,3:4],xy[I,4:5],xy[I,2:3]),axis=1)    

xtmp=xy[I,0]
ytmp=xy[I,1]
p=xy[I,2]
u=xy[I,3]
v=xy[I,4]

#load model
#session-run
tf.reset_default_graph    
graph = tf.get_default_graph() 
#load model
with tf.Session() as sess1:
    
    path1='./tf_model/case_1_re%s_nodp_nodv/tf_model/'%Re
    new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
    new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))

    tf_dict = {'input1a:0': xtmp[:,None], 'input1b:0': ytmp[:,None], \
               'input1c:0': ytmp[:,None]/ytmp.max(), 'input1d:0': ytmp[:,None]/ytmp.max() }

    op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
    
    #uvp
    tout = sess1.run(op_to_load1, tf_dict)

sess1.close()

p_pred=tout[:,2]
u_pred=tout[:,0]
v_pred=tout[:,1]


    
    

###########################################################################
###-------------Error-------------------------


  

    
tmp=p-p_pred
train_l2_p = (LA.norm(tmp)/LA.norm(p))*100 

tmp=u-u_pred
train_l2_u=(LA.norm(tmp)/LA.norm(u))*100 

tmp=v-v_pred
train_l2_v=(LA.norm(tmp)/LA.norm(v))*100     
    
   
    
print('error', (train_l2_u + train_l2_v)*0.5 )    
    
