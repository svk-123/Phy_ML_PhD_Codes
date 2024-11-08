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
import pickle
import tensorflow as tf


import keras
from keras.models import load_model
import shutil

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""

# read data from below dir...
path='./case'

indir = path

fname_1 = [f for f in listdir(indir) if isdir(join(indir, f))]
fname_1.sort()
fname_1=np.asarray(fname_1)

#np.random.seed(1234)
#np.random.shuffle(fname)

fname_2=[]
for i in range(len(fname_1)):
    dir2=indir + '/%s'%fname_1[i]
    tmp=[f for f in listdir(dir2) if isdir(join(dir2, f))]
    fname_2.append(tmp)
fname_2=fname_2[0]
fname_2=np.asarray(fname_2)

reno=[]
for i in range(len(tmp)):
    reno.append(tmp[i].split('_')[1])    

reno=np.array(reno)
reno = reno.astype(np.float)

            #load model
            #session-run
tf.reset_default_graph    
graph = tf.get_default_graph() 


#no use loop
for jj in range(1):
  
    for ii in range(1):
        
        print ( ii)
        
        casedir= path +'/%s/%s'%(fname_1[jj],fname_2[ii])
        print(casedir)
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname[:-3].astype(np.float) 
                
        xx=np.loadtxt(casedir+'/postProcessing/forceCoeffs/1/forceCoeffs.dat', skiprows=10)
        xx=xx[::2,:]
        xx=xx[-60:]
        xx=xx[xx[:,3].argsort()]
        
#       plt.figure(figsize=(10, 4))
#       plt.plot(xx[:,0],xx[:,3],'ob')
#       plt.plot([t1,t10],[xx[:,3].mean(),xx[:,3].mean()],'or')
#       plt.savefig('./plots/%s.png'%ii,format='png',dpi=100)
#       plt.close()
            
        t1=xx[0,0]
        t2=xx[1,0]
        
        if (abs(xx[0,0]-xx[1,0]) > 6):
            t2=xx[2,0]
            
        if (t1 > t2):
            tmp1= t1
            t1 =t2
            t2 = tmp1
   
        tt = np.linspace(t1,t2,int (round((t2-t1)/0.2)+1) )
         
        tt=tt[:-1]
        mytt = tt-t1
        #mytt = mytt/mytt.max()
                                      
        for kk in [0,8,16]:  
            
            ymax=round(tt[kk],2)
            if((ymax%1) == 0):
                ymax=int(ymax)
            print ('t = ', ymax)
            
            x=[]
            with open(casedir +'/%s/ccx'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    x.append(line)
            x=np.array(x)        
            x = x.astype(np.float)
           
            y=[]
            with open(casedir +'/%s/ccy'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    y.append(line)
            y=np.array(y)        
            y = y.astype(np.float)
            
            z=[]
            with open(casedir +'/%s/ccz'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    z.append(line)
            z=np.array(z)        
            z = z.astype(np.float)
            
            p=[]
            with open(casedir +'/%s/p'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    p.append(line)
            p=np.array(p)        
            p = p.astype(np.float)
            
            
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
                    
            u=np.array(u)        
            u = u.astype(np.float)
            v=np.array(v)        
            v = v.astype(np.float)               
            w=np.array(w)        
            w = w.astype(np.float)       
                              
            #filter within xlim,ylim
            I=[]
            for i in range(len(x)):
                if (x[i]<=2 and x[i]>=-2 and y[i]<=2 and y[i]>=-2):
                    I.append(i)
                              
            xl=x[I]
            yl=y[I]
            zl=z[I]
            ul=u[I]
            vl=v[I]
            wl=w[I]
            pl=p[I]
             
            tlist=[]
            for k in range(len(xl)):
                tlist.append(mytt[kk])
            tlist=np.asarray(tlist)
                                   
            relist=[]
            for k in range(len(xl)):
                relist.append(reno[ii])
            relist=np.asarray(relist)
                     
            #load model
            with tf.Session() as sess1:
                
                path1='./tf_model/case_1_around_100x8_5k_15k/tf_model/'
                new_saver1 = tf.train.import_meta_graph( path1 + 'model_0.meta')
                new_saver1.restore(sess1, tf.train.latest_checkpoint(path1))
                
                tf_dict = {'input0:0': xl[:,None], 'input1:0': yl[:,None], 'input2:0': tlist[:,None]}
                
                op_to_load1 = graph.get_tensor_by_name('NS1/prediction/BiasAdd:0')    
                
                #uvp
                out = sess1.run(op_to_load1, tf_dict)
            sess1.close()  
            
            #plot
            def plot(xp,yp,zp,nc,name):
                plt.figure(figsize=(3, 4))
                #cp = pyplot.tricontour(ys, zs, pp,nc)
                cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
                #cp=pyplot.tricontourf(x1,y1,z1)
                #cp=pyplot.tricontourf(x2,y2,z2)   
                
                #cp = pyplot.tripcolor(xp, yp, zp)
                #cp = pyplot.scatter(ys, zs, pp)
                #pyplot.clabel(cp, inline=False,fontsize=8)
                #plt.xlim(-1,2)
                #plt.ylim(-1,1)    
                #plt.axis('off')
                #plt.grid(True)
                #patch.set_facecolor('black')
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                #plt.savefig('./plotc/%s.eps'%(nname[ii]), format='eps')
                plt.show()
                
            #plot(xl,yl,out[:,0],20,'name') 
            #plot(xl,yl,pl,20,'name') 
            
            # p- write
            p=[]
            with open(casedir +'/%s/p'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    p.append(line)
            p=np.array(p)        
            p = p.astype(np.float)
        
            p[I]=out[:,0]
                        
            dst2='./case_ml' +'/%s/%s'%(fname_2[ii],ymax)
            
            if os.path.exists(dst2):
                shutil.rmtree(dst2)
                        
            shutil.copytree(casedir+'/%s'%ymax,dst2)
            
            if(kk==0):
                
                dst11='./case_ml' + '/%s/0'        %(fname_2[ii])
                dst12='./case_ml' + '/%s/constant' %(fname_2[ii])
                dst13='./case_ml' + '/%s/system'   %(fname_2[ii])
                
                if os.path.exists(dst11):
                    shutil.rmtree(dst11)
                if os.path.exists(dst12):
                    shutil.rmtree(dst12)
                if os.path.exists(dst13):
                    shutil.rmtree(dst13)
                    
                shutil.copytree(casedir + '/0'        , dst11 )                
                shutil.copytree(casedir + '/constant' , dst12 )            
                shutil.copytree(casedir + '/system'   , dst13 )           
            
            print ('writing-p')
            fp= open(dst2 +'/p', 'w+')
            
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
            w=[]
            with open(casedir +'/%s/U'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    line=line.replace("(","")
                    line=line.replace(")","")        
                    a, b, c = (item.strip() for item in line.split(' ', 3))
                    u.append(a), v.append(b), w.append(c)
                    
            u=np.array(u)        
            u = u.astype(np.float)
            v=np.array(v)        
            v = v.astype(np.float)               
            w=np.array(w)        
            w = w.astype(np.float) 

            u[I]=out[:,1]
            v[I]=out[:,2]                      
            
            print ('writing-U')
            fp= open(dst2 +'/U', 'w+')
            
            for i in range(22):
                fp.write("%s"%(data0[i]))
            for i in range(npt):
                fp.write("(%f %f 0.0)\n"%(u[i],v[i]))
            for i in range((22+npt),len(data0)):    
                fp.write("%s"%(data0[i]))        
            fp.close()


      


    
      
