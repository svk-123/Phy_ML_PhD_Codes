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
import pickle
import pandas
from scipy.interpolate import griddata
from scipy import interpolate
from numpy import linalg as LA
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

#load data
inp_x=[]
inp_y=[]
inp_reno=[]
inp_aoa=[]
inp_para=[]
inp_t=[]

out_p=[]
out_u=[]
out_v=[]

#load data
#with open('./data_file/ph_1_test/foil_aoa_nn_p16_ph_1_ts_1.pkl', 'rb') as infile:
with open('./data_file/foil_naca_un_turb_ts_1.pkl', 'rb') as infile:
    result = pickle.load(infile)

inp_x.extend(result[0])   
inp_y.extend(result[1])
inp_para.extend(result[2])
inp_reno.extend(result[3])
inp_aoa.extend(result[4])
inp_t.extend(result[5])

out_p.extend(result[6])
out_u.extend(result[7])
out_v.extend(result[8])

co=result[9]
#fxy=result[9]
name=result[10]

tt=result[11]

inp_x=np.asarray(inp_x)
inp_y=np.asarray(inp_y)
inp_reno=np.asarray(inp_reno)
inp_aoa=np.asarray(inp_aoa)
inp_para=np.asarray(inp_para)
inp_t=np.asarray(inp_t)

out_p=np.asarray(out_p)
out_u=np.asarray(out_u)
out_v=np.asarray(out_v)

cotmp=[]
for i in range(len(co)):
    for j in range(5):
        cotmp.append(co[i])
co=cotmp        


#plot
def stream_plot(i):
    
    fig = plt.figure(figsize=(10, 3),dpi=100)
    
    pts_x=np.linspace(0,1,50)
    pts_y=np.linspace(0.0,0.5,50)
    xx,yy=np.meshgrid(pts_x,pts_y)
    
    pts=np.concatenate((xx.flatten()[:,None],yy.flatten()[:,None]),axis=1)
    
    points=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()
    grid_y, grid_x = np.mgrid[-0.5:0.5:200j, -0.5:2:200j]
    u1 = griddata(points, val_out[:,1], (grid_x, grid_y), method='linear')
    v1 = griddata(points, val_out[:,2], (grid_x, grid_y), method='linear')

    u2 = griddata(points, out[:,1], (grid_x, grid_y), method='linear')
    v2 = griddata(points, out[:,2], (grid_x, grid_y), method='linear')
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.streamplot(grid_x,grid_y,u1,v1,density=2,linewidth=0.5,color=u1, cmap=cm.jet,arrowsize=0.02,\
                   minlength=0.2, maxlength=4.0, zorder=0)
    ax1.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='gray',zorder=5)
    #ax1.set_title('CFD: t=T/5')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    ax1.set_xlim([-0.5, 2])
    ax1.set_ylim([-0.5, 0.5])
    plt.subplots_adjust( wspace=0.2,hspace=0.3)
    ax1.set_aspect(1.1)
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.streamplot(grid_x,grid_y,u2,v2,density=2,linewidth=0.5,color=u1, cmap=cm.jet,arrowsize=0.02,\
                   minlength=0.2, maxlength=4.0,zorder=0)
    ax2.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='gray',zorder=5)
    #ax2.set_title('MLP: t=T/5')
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_ylabel('Y',fontsize=16)
    ax2.set_xlim([-0.5, 2])
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_aspect(1.1)
   
#    
#    ax3 = fig.add_subplot(3,2,3)
#    cp3 = ax3.tricontourf(val_inp[:,0],val_inp[:,1],val_out[:,1],20,cmap=cm.jet)
#    ax3.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
#    ax3.set_title('u-cfd')
#    ax3.set_xlabel('X',fontsize=16)
#    ax3.set_ylabel('Y',fontsize=16)
#    ax3.set_xlim([-0.5, 2])
#    ax3.set_ylim([-0.5, 0.5])
#    divider = make_axes_locatable(ax3)
#    cax = divider.append_axes('right', size='5%', pad=0.05)
#    cbar3=plt.colorbar(cp3, cax=cax, orientation='vertical');
#    cbar3.ax.tick_params(labelsize=10)
#    ax3.set_aspect(1.1)
#    
#    
#    ax4 = fig.add_subplot(3,2,4)
#    cp4 = ax4.tricontourf(val_inp[:,0],val_inp[:,1],out[:,1],20,cmap=cm.jet)
#    ax4.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
#    ax4.set_title('u-NN')
#    ax4.set_xlabel('X',fontsize=16)
#    ax4.set_yticks([])
#    ax4.set_xlim([-0.5, 2])
#    ax4.set_ylim([-0.5, 0.5])
#    divider = make_axes_locatable(ax4)
#    cax = divider.append_axes('right', size='5%', pad=0.05)
#    cbar4=plt.colorbar(cp4, cax=cax, orientation='vertical');
#    cbar4.ax.tick_params(labelsize=10)
#    ax4.set_aspect(1.1)      
#    
#    
#    ax5 = fig.add_subplot(3,2,5)
#    cp5 = ax5.tricontourf(val_inp[:,0],val_inp[:,1],val_out[:,2],20,cmap=cm.jet)
#    ax5.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
#    ax5.set_title('v-cfd')
#    ax5.set_xlabel('X',fontsize=16)
#    ax5.set_ylabel('Y',fontsize=16)
#    ax5.set_xlim([-0.5, 2])
#    ax5.set_ylim([-0.5, 0.5])
#    divider = make_axes_locatable(ax5)
#    cax = divider.append_axes('right', size='5%', pad=0.05)
#    cbar5=plt.colorbar(cp5, cax=cax, orientation='vertical');
#    cbar5.ax.tick_params(labelsize=10)
#    ax5.set_aspect(1.1)
#    
#    
#    ax6 = fig.add_subplot(3,2,6)
#    cp6 = ax6.tricontourf(val_inp[:,0],val_inp[:,1],out[:,2],20,cmap=cm.jet)
#    ax6.tricontourf(co[i][:,0],co[i][:,1],np.zeros(len(co[i])),colors='w')
#    ax6.set_title('v-NN')
#    ax6.set_xlabel('X',fontsize=16)
#    ax6.set_yticks([])
#    ax6.set_xlim([-0.5, 2])
#    ax6.set_ylim([-0.5, 0.5])
#    divider = make_axes_locatable(ax6)
#    cax = divider.append_axes('right', size='5%', pad=0.05)
#    cbar6=plt.colorbar(cp6, cax=cax, orientation='vertical');
#    cbar6.ax.tick_params(labelsize=10)
#    ax6.set_aspect(1.1)
    
    #fig.suptitle("Testing:Airfoil-%s Re=%se3 AoA=%s"%(name[i][0],int(inp_reno[i][0]*10),int(inp_aoa[i][0]*20)), fontsize=24)
    
    plt.subplots_adjust(top = 0.9, bottom = 0.25, right = 0.98, left = 0.1, hspace = 0.2, wspace = 0.3)   
    plt.savefig('./plot/%s_%s_Re=%s_AoA=%s.png'%(i,name[i][0],int(inp_reno[i][0]*10000),int(inp_aoa[i][0]*20)),format='png',dpi=200)
    plt.close()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
np.random.seed(1234534)
mylist=[5,6,7,8,9]

#load_model
model_test=load_model('./selected_model/case_2_8x500/model_sf_1000_0.00005640_0.00005714.hdf5') 
for i in mylist:
    print (i)
    #normalize
    inp_reno[i]=inp_reno[i]/10000.
    inp_aoa[i]=inp_aoa[i]/20.0
    
    val_inp=np.concatenate((inp_x[i][:,None],inp_y[i][:,None],inp_reno[i][:,None],inp_aoa[i][:,None],inp_t[i][:,None],inp_para[i][:,:]),axis=1)
    val_out=np.concatenate((out_p[i][:,None],out_u[i][:,None],out_v[i][:,None]),axis=1)


    out=model_test.predict([val_inp]) 
    
    #stream_plot(i)     
    
    



