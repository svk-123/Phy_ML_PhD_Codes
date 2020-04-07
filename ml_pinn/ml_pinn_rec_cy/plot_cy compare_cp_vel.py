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
from os.path import isfile, join, isdir

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
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

#load data
xtmp=[]
ytmp=[]
p=[]
u=[]
v=[]
p_pred=[]
u_pred=[]
v_pred=[]


for ii in range(1):
    #x,y,Re,u,v
    with open('./data_file/cy_40_around_2222.pkl', 'rb') as infile:
        result = pickle.load(infile)
    xtmp.extend(result[0])
    ytmp.extend(result[1])
    p.extend(result[2])
    u.extend(result[3])
    v.extend(result[4])    

    
xtmp=np.asarray(xtmp)
ytmp=np.asarray(ytmp)
p=np.asarray(p)
u=np.asarray(u)
v=np.asarray(v)

co=np.zeros((100,2))
theta=np.linspace(0,360,100)
for i in range(len(theta)):
    co[i,0]= 0.5*np.cos(np.radians(theta[i]))
    co[i,1]= 0.5*np.sin(np.radians(theta[i]))  
    
val_inp=np.concatenate((xtmp[:,None],ytmp[:,None]),axis=1)
val_out=np.concatenate((u[:,None],v[:,None],p[:,None]),axis=1)    


#....................................................................

def plot_cp(myp):
        
    #LinearNDinterpolator
    pD=np.asarray([xtmp,ytmp]).transpose()

    a0=50
    
    xu=co[:a0+1,0]
    yu=co[:a0+1,1]

        
    xl=co[a0:,0]
    yl=co[a0:,1]
 
             

    #for -p
    print ('interpolation-1...')      
    f1p=interpolate.LinearNDInterpolator(pD,myp)
        
    pu1=np.zeros(len(xu))
    for j in range(len(xu)):
        pu1[j]=f1p(xu[j],yu[j])
    pl1=np.zeros(len(xl))
    for j in range(len(xl)):
        pl1[j]=f1p(xl[j],yl[j])
        
   
    return (xu,pu1)
####---------------------------------------------------------------------------------------

    
#plot
def plot_uv(u_pred,v_pred):


    #LinearNDinterpolator
    pD=np.asarray([val_inp[:,0],val_inp[:,1]]).transpose()

    
    xa=np.linspace(-0.25,-0.25,50)
    ya=np.linspace(0,0.5,50)

    xb=np.linspace(0,0,50)
    yb=np.linspace(0.5,1.0,50)

    xc=np.linspace(1,1,50)
    yc=np.linspace(-0,1,50)

    xd=np.linspace(1.5,1.5,50)
    yd=np.linspace(-0,1,50)
        
        
    # for u    
    print 'interpolation-1...'      
    f1u=interpolate.LinearNDInterpolator(pD,u_pred)
        
    u1a=np.zeros((len(ya)))
    u1b=np.zeros((len(ya)))
    u1c=np.zeros((len(ya)))
    u1d=np.zeros((len(ya)))
    for j in range(len(ya)):
        u1a[j]=f1u(xa[j],ya[j])
        u1b[j]=f1u(xb[j],yb[j])
        u1c[j]=f1u(xc[j],yc[j])
        u1d[j]=f1u(xd[j],yd[j])
        


    #for -v
    print 'interpolation-1...'      
    f1v=interpolate.LinearNDInterpolator(pD,v_pred)

    v1a=np.zeros((len(ya)))
    v1b=np.zeros((len(ya)))
    v1c=np.zeros((len(ya)))
    v1d=np.zeros((len(ya)))
    for j in range(len(ya)):
        v1a[j]=f1v(xa[j],ya[j])
        v1b[j]=f1v(xb[j],yb[j])
        v1c[j]=f1v(xc[j],yc[j])
        v1d[j]=f1v(xd[j],yd[j])

    b=np.asarray([yb,u1b,v1b])
    c=np.asarray([yc,u1c,v1c])
    d=np.asarray([yd,u1d,v1d])
    return (b,c,d)
    
 






#####------------------------------------------------------------------------------------
CP=[]
B1=[]
C1=[]
D1=[]

path='./tf_model'
tmp=[f for f in listdir(path) if isdir(join(path, f))]
case_dir=np.asarray(tmp)
case_dir.sort()

mylabel=['CFD','PINN_2222_3S_around_80','PINN_2222_3S_wake_80','PINN_2222_4S_around_80','PINN_5555_4S_around_80']

#for CFD data
p_cfd=plot_cp(p)
CP.append(p_cfd)

b1,c1,d1=plot_uv(u,v)
B1.append(b1)
C1.append(c1)
D1.append(d1)

for jj in range(0,4):

    #load model
    #session-run
    tf.reset_default_graph    
    graph = tf.get_default_graph() 
    #load model
    with tf.Session() as sess1:
        
        path1='./tf_model/%s/tf_model/'%case_dir[jj]
        new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
        new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))
    
        tf_dict = {'input0:0': xtmp[:,None], 'input1:0': ytmp[:,None]}
    
        op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
        
        #uvp
        tout = sess1.run(op_to_load1, tf_dict)
    
    sess1.close()
    
    
    p_pred=tout[:,2]
    u_pred=tout[:,0]
    v_pred=tout[:,1]
    
    
    p_interp=plot_cp(p_pred)
    
    CP.append(p_interp)
    
    b1,c1,d1=plot_uv(u_pred,v_pred)
    
    B1.append(b1)
    C1.append(c1)
    D1.append(d1)
   



#plot CP distrubution 
    
c=['g','b','y','c','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown'] 
mei=2       
plt.figure(figsize=(6,5),dpi=100)
plt.plot(CP[0][0],CP[0][1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
for jj in range(1,5):
    plt.plot(CP[jj][0],CP[jj][1],'%s'%c[jj],linewidth=3,label='%s'%mylabel[jj])
#plt.plot(xl,pl2,'k',linewidth=3,label='NN-lower')  
plt.xlabel('X',fontsize=20)
plt.ylabel('P',fontsize=20)
plt.title('Pressure Dist. over cylinder')
plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.75), ncol=1, frameon=True, fancybox=False, shadow=False)
plt.savefig('./plot/cp_compare.png',format='png',bbox_inches='tight', dpi=100)
plt.show()



# plot velocity profile - u
mei=2
#UV [case][y,u,v]   
plt.figure(figsize=(6, 4), dpi=100)

plt.subplot(1,3,1)
plt.plot(B1[0][1,:],B1[0][0,:],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei)

for i in range(1,5):
    plt.plot(B1[i][1,:],B1[i][0,:],'%s'%c[i],lw=3)
    
plt.ylabel('Y',fontsize=20)
#plt.xlim(-0.1,1.2)
plt.ylim(-0.05,1.05)    
    
plt.subplot(1,3,2)
plt.plot(C1[0][1,:],C1[0][0,:],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei)

for i in range(1,5):
    plt.plot(C1[i][1,:],C1[i][0,:],'%s'%c[i],lw=3)
    
plt.yticks([]) 
plt.xlabel('U',fontsize=20)
#plt.xlim(-0.1,1.2)
plt.ylim(-0.05,1.05)   
        
plt.subplot(1,3,3)
plt.plot(D1[0][1,:],D1[0][0,:],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')

for i in range(1,5):
    plt.plot(D1[i][1,:],D1[i][0,:],'%s'%c[i],lw=3,label='%s'%mylabel[i])
    
plt.yticks([]) 
#plt.xlim(-0.1,1.2)
plt.ylim(-0.05,1.05)      
    
plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.6), ncol=1, frameon=True, fancybox=False, shadow=False)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/u_comb.png', format='png', bbox_inches='tight',dpi=100)
plt.show()   
plt.close()
    
    
# plot velocity profile - v
mei=2
#UV [case][y,u,v]   
plt.figure(figsize=(6, 4), dpi=100)

plt.subplot(1,3,1)
plt.plot(B1[0][2,:],B1[0][0,:],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei)

for i in range(1,5):
    plt.plot(B1[i][2,:],B1[i][0,:],'%s'%c[i],lw=3)
    
plt.ylabel('Y',fontsize=20)
#plt.xlim(-0.1,1.2)
plt.ylim(-0.05,1.05)    
    
plt.subplot(1,3,2)
plt.plot(C1[0][2,:],C1[0][0,:],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei)

for i in range(1,5):
    plt.plot(C1[i][2,:],C1[i][0,:],'%s'%c[i],lw=3)
    
plt.yticks([]) 
plt.xlabel('V',fontsize=20)
#plt.xlim(-0.1,1.2)
plt.ylim(-0.05,1.05)   
        
plt.subplot(1,3,3)
plt.plot(D1[0][2,:],D1[0][0,:],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')

for i in range(1,5):
    plt.plot(D1[i][2,:],D1[i][0,:],'%s'%c[i],lw=3,label='%s'%mylabel[i])
    
plt.yticks([]) 
#plt.xlim(-0.1,1.2)
plt.ylim(-0.05,1.05)      
    
plt.figtext(0.4, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)
plt.legend(loc='center left', fontsize=18, bbox_to_anchor=(1,0.6), ncol=1, frameon=True, fancybox=False, shadow=False)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/v_comb.png', format='png', bbox_inches='tight',dpi=100)
plt.show()   
plt.close()    



