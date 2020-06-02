#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:49:29 2017

This code process OF data and exports as .pkl to prepData file
for TBNN. prepData reads .pkl and process further

@author: vino
"""
# imports
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy import interpolate
from os import listdir
from os.path import isfile,isdir, join
import cPickle as pickle
import tensorflow as tf
import shutil

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""


fname_1=['foil']

fname_1=np.asarray(fname_1)
fname_1.sort()


# read data from below dir...
path='./foil'
#path='/home/vino/ml_test/from_nscc_26_dec_2018/foam_run/case_naca_lam'
indir = path



#np.random.seed(1234)
#np.random.shuffle(fname)

fname_2=[]
#for i in range(len(fname_1)):
#    dir2=indir + '/%s'%fname_1[i]
#    tmp=[f for f in listdir(dir2) if isdir(join(dir2, f))]
#    fname_2.append(tmp)

#################
'''chose -a irfoil to process'''

#fname_2.append('naca2412_000100_00')
fname_2.append(['naca4412_000100_20'])  
    


tmp=[]
foil=[]
for i in range(len(fname_1)):
    for j in range(len(fname_2[i])):
        tmp.append(fname_2[i][j])
        foil.append(fname_2[i][j].split('_')[0])
tmp=np.asarray(tmp)    
foil=np.asarray(foil)

       
coord=[]
#for nn in range(len(foil)):
#    pts=np.loadtxt('./coord_seligFmt_formatted/%s.dat'%foil[nn],skiprows=1)
#    coord.append(pts)
# 


aoa=[]
reno=[]
for i in range(len(tmp)):
    reno.append(tmp[i].split('_')[1])    
    aoa.append(tmp[i].split('_')[2])

reno=np.array(map(float, reno))
aoa = np.array(map(float, aoa))


st= [0]
end=[1]

fp=open('foil_error.dat','w+')
fp1=open('foil_details.dat','w+')

'''np.random.seed(1234534)
mylist=np.random.randint(0,4500,100)

nco=[]
for k in mylist:
    nco.append(coord[k])'''


for jj in range(1):

    myinp_x=[]
    myinp_y=[]
    myinp_para=[]
    myinp_aoa=[]
    myinp_re=[]

    myout_p=[]
    myout_u=[]
    myout_v=[]

    myname=[]
    myfname=[]

    #for ii in mylist:
    for ii in range(st[jj],end[jj]):
        print (ii)
        
        casedir= path +'/%s'%(tmp[ii])
                
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname[:-3].astype(np.int) 
        ymax=int(yname.max())

        fp1.write('%s-%s\n'%(ii,casedir))  
        fp1.write('	yname:%s\n'%(yname))
        fp1.write('	ymax:%s\n'%(ymax)) 

        x=[]
        with open(casedir +'/%s/ccx'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                x.append(line)
        x = np.array(map(float, x))
       
        y=[]
        with open(casedir +'/%s/ccy'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                y.append(line)
        y = np.array(map(float, y))
        
        z=[]
        with open(casedir +'/%s/ccz'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                z.append(line)
        z = np.array(map(float, z))
        
        p=[]
        with open(casedir +'/%s/p'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                p.append(line)
        p = np.array(map(float, p))
        
        
        # load velocity
        u=[]
        v=[]
        w=[]
        with open(casedir +'/%s/U'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c = (item.strip() for item in line.split(' ', 3))
                u.append(a), v.append(b), w.append(c)
        u = np.array(map(float, u))
        v = np.array(map(float, v))
        w = np.array(map(float, w))
               
        #filter within xlim,ylim
        I=[]
        for i in range(len(x)):
            if (x[i]<=2 and x[i]>=-1 and y[i]<=1 and y[i]>=-1 ):
                I.append(i)
        xl=x[I]
        yl=y[I]
        zl=z[I]
        
        #load model
        #session-run
        tf.reset_default_graph    
        graph = tf.get_default_graph() 
        #load model
        with tf.Session() as sess1:
            
            path1='./tf_model/case_1_naca4412_Re100_aoa20_nodp_nodv_ws_ar_30x2/tf_model/'
            new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
            new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))
        
            tf_dict = {'input1a:0': xl[:,None], 'input1b:0': yl[:,None], \
               'input1c:0': yl[:,None]/yl.max(), 'input1d:0': yl[:,None]/yl.max() }
        
            op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
            
            #uvp
            out = sess1.run(op_to_load1, tf_dict)
        
        sess1.close()
        
        # p- write
        p=[]
        with open(casedir +'/%s/p'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                p.append(line)
        p = np.array(map(float, p))
        
        pl=p[I].copy()
        p[I]=out[:,2]
        
                
        dst2='./foil_ml/%s_pinn_dp'%(tmp[ii])
        
        if os.path.exists(dst2):
            shutil.rmtree(dst2)
                    
        shutil.copytree(casedir,dst2)
        
        print 'writing-p'
        fp= open(dst2 +'/%s/p'%ymax, 'w+')
        
        for i in range(22):
            fp.write("%s"%(data0[i]))
        for i in range(npt):
            fp.write("%f\n"%(p[i]))
        for i in range((22+npt),len(data0)):    
            fp.write("%s"%(data0[i]))        
        fp.close() 
             

        # load velocity
        u=[]
        v=[]
        with open(casedir +'/%s/U'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c = (item.strip() for item in line.split(' ', 3))
                u.append(a), v.append(b)
        u = np.array(map(float, u))
        v = np.array(map(float, v))

        u[I]=out[:,0]
        v[I]=out[:,1]                      
        

        print 'writing-U'
        fp= open(dst2 +'/%s/U'%ymax, 'w+')
        
        for i in range(22):
            fp.write("%s"%(data0[i]))
        for i in range(npt):
            fp.write("(%f %f 0.0)\n"%(u[i],v[i]))
        for i in range((22+npt),len(data0)):    
            fp.write("%s"%(data0[i]))        
        fp.close() 

