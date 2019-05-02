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
path='./case_foam'

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
    
tmp=[]
foil=[]
for i in range(len(fname_1)):
    for j in range(len(fname_2[i])):
        tmp.append(fname_2[i][j])
        foil.append(fname_2[i][j].split('_')[0])
tmp=np.asarray(tmp)    
foil=np.asarray(foil)


tmp=[]
foil=[]
for i in range(len(fname_1)):
    for j in range(len(fname_2[i])):
        tmp.append(fname_2[i][j])
        foil.append(fname_2[i][j].split('_')[0])
tmp=np.asarray(tmp)    
foil=np.asarray(foil)
       
coord=[]
for nn in range(len(foil)):
    pts=np.loadtxt('../cnn_airfoil_sf/airfoil_data/coord_seligFmt_formatted/%s.dat'%foil[nn],skiprows=1)
    coord.append(pts)
 
datafile='./data_file/naca4_digit_para_opti_foil.pkl'
with open(datafile, 'rb') as infile:
    result = pickle.load(infile)
para=result[0]   
pname=result[3]
para=np.asarray(para)
pname=np.asarray(pname)

aoa=[]
reno=[]
for i in range(len(tmp)):
    reno.append(tmp[i].split('_')[1])    
    aoa.append(tmp[i].split('_')[2])

reno=np.array(reno)
aoa = np.array(aoa)

reno = reno.astype(np.float)
aoa  = aoa.astype(np.float)

my_para=[]
for i in range(len(foil)):
    if foil[i] in pname:
        ind=np.argwhere(pname==foil[i])
        my_para.append(para[int(ind)])

    else:
        print('not in pname %s'%foil[i])


#no use loop
for jj in range(1):

    for ii in range(1):
        print (ii)
        
        
        
        casedir= path +'/%s/%s'%(foil[ii],tmp[ii])
                
        #need to find max time later....
#        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
#        yname = np.asarray(yname)
#        yname.sort()
#        yname=yname[:-3].astype(np.int) 
#        ymax=int(yname.max())
        
        ymax=0.25
        for kk in range(1):  
            
            x=[]
            with open(casedir +'/%s/Cx'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    x.append(line)
            x=np.array(x)        
            x = x.astype(np.float)
           
            y=[]
            with open(casedir +'/%s/Cy'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    y.append(line)
            y=np.array(y)        
            y = y.astype(np.float)
            
            z=[]
            with open(casedir +'/%s/Cz'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    z.append(line)
            z=np.array(z)        
            z = z.astype(np.float)
            
         

            
            #filter within xlim,ylim
            I=[]
            for i in range(len(x)):
                if (x[i]<=2.2 and x[i]>=-0.6 and y[i]<=0.6 and y[i]>=-0.6 ):
                    I.append(i)
            xl=x[I]
            yl=y[I]
            zl=z[I]
            
            relist=[]
            for k in range(len(xl)):
                relist.append(reno[ii])
            relist=np.asarray(relist)    
            
     
            aoalist=[]
            for k in range(len(xl)):
                aoalist.append(aoa[ii])
            aoalist=np.asarray(aoalist) 
            
            paralist=[]
            for k in range(len(xl)):
                paralist.append(my_para[jj])
            paralist=np.asarray(paralist) 
            
            tlist=[]
            for k in range(len(xl)):
                tlist.append(ymax)
            tlist=np.asarray(tlist) 
            
            #ml-predict        
            relist=relist/10000.
            aoalist=aoalist/20.
            val_inp=np.concatenate((xl[:,None],yl[:,None],relist[:,None],aoalist[:,None],tlist[:,None],paralist[:,:]),axis=1)
            
     
            model_test=load_model('./selected_model/case_2_8x500/model_sf_1000_0.00005640_0.00005714.hdf5')
            out=model_test.predict([val_inp])
        
            # p- write
            p=[]
            with open(casedir +'/%s/p'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    p.append(line)
            p=np.array(p)        
            p = p.astype(np.float)
        
            pl=p[I].copy()
            p[I]=out[:,0]
        
                
            dst2='./case_ml' +'/%s/%s/%s'%(foil[ii],tmp[ii],ymax)
            
            if os.path.exists(dst2):
                shutil.rmtree(dst2)
                        
            shutil.copytree(casedir+'/%s'%ymax,dst2)
            
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

        


    
      
