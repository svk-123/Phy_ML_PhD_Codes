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
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import random
import cPickle as pickle
import os, shutil

#rbf commons
#layer-1 centres
with open('./rbfcom/centers_1.pkl', 'rb') as infile:
    result = pickle.load(infile)
print('found centers')
c1=result[0]

#layer-2 centers
c2=np.asarray([0.1,0.2,0.3,0.5,1.0])
c2=np.reshape(c2,(5,1))

sp1=0.2
sp2=0.4

def fit_layer1(mylist):
    
    flist=mylist
    #layer-1
    for ii in range(len(flist)):
        
        #x,y,Re,u,v    
        xtmp=[]
        ytmp=[]
        reytmp=[]
        utmp=[]
        vtmp=[]
        #load data
        with open('./data/cavity_%s.pkl'%flist[ii], 'rb') as infile:
            result = pickle.load(infile)
        xtmp.extend(result[0])
        ytmp.extend(result[1])
        reytmp.extend(result[2])
        utmp.extend(result[3])
        vtmp.extend(result[4])
        
        xtmp=np.asarray(xtmp)
        ytmp=np.asarray(ytmp)
        reytmp=np.asarray(reytmp)
        utmp=np.asarray(utmp)
        vtmp=np.asarray(vtmp)
        
        # ---------ML PART:-----------#
        #shuffle data
        N= len(utmp)
        I = np.arange(N)
        np.random.shuffle(I)
        n=10000
        
        #normalize
        reytmp=reytmp/1000.
        
        my_inp=np.concatenate((xtmp[:,None],ytmp[:,None],reytmp[:,None]),axis=1)
        my_out=np.concatenate((utmp[:,None],vtmp[:,None]),axis=1)

        
        x=my_inp[:,0:2]
        y=my_out[:,0]
        c=c1
        
        from mrbf_layers import layer_1
        l1u=layer_1(x,y,c,x.shape[0],c.shape[0],2,sp1)
        l1u.f_mq()
        l1u.ls_solve()
        l1u.pred_f_mq()
        predu=l1u.pred
    
        x=my_inp[:,0:2]
        y=my_out[:,1]
    
        l1v=layer_1(x,y,c,x.shape[0],c.shape[0],2,sp1)
        l1v.f_mq()
        l1v.ls_solve()
        l1v.pred_f_mq()
        predv=l1v.pred
    
        def plot1(x,y,name):
            plt.plot(x, y, 'o', label='%s'%name)
            plt.plot([-0.65,1.6],[-0.65,1.6] ,'r')
            plt.xlabel('true',fontsize=16)
            plt.ylabel('pred',fontsize=16)
            plt.legend(fontsize=16)
            plt.savefig('%s'%name,format='png', dpi=100)
            plt.show()
    
        plot1(my_out[:,0],predu,'rbf-u-%s'%flist[ii])
        plot1(my_out[:,1],predv,'rbf-v-%s'%flist[ii])
    
        #pickle 
        data_rbf=[l1u.w,l1v.w]
        #weights for each Re- cavity
        with open('./rbfout_1/cavity_weight1_w%s_%s.pkl'%(c.shape[0],flist[ii]), 'wb') as outfile:
            pickle.dump(data_rbf, outfile, pickle.HIGHEST_PROTOCOL)


#flist=['Re100','Re200','Re300','Re500','Re1000']
#fit_layer1(flist)

 
def fit_layer2(mylist,myreno):

    flist=mylist
    reno=myreno
    
    #layer-2    
    wtmp_u=[]
    wtmp_v=[]
    reytmp=[]
    for ii in range(len(flist)):
        #wu,wv
        #load data
        with open('./rbfout_1/cavity_weight1_w200_%s.pkl'%flist[ii], 'rb') as infile:
            result = pickle.load(infile)
        wtmp_u.append(result[0])
        wtmp_v.append(result[1])
        reytmp.append(np.full(200,reno[ii]))
    
    wtmp_u=np.asarray(wtmp_u)
    wtmp_v=np.asarray(wtmp_v)
    reytmp=np.asarray(reytmp)
    
    #normalize
    wtmp=wtmp_u/10000.
    reytmp=reytmp/1000.
    
    my_inp=reytmp
    my_out=wtmp
    
    x=my_inp.copy()
    y=my_out.copy()
    c=c2   
  
    from mrbf_layers import layer_2
    l2u=layer_2(x,y,c,x.shape[0],c.shape[0],1,sp2)
    l2u.f_mq()
    l2u.ls_solve()
    l2u.pred_f_mq()
    predu=l2u.pred
    
    #normalize
    wtmp=wtmp_v/10000.
    
    my_inp=reytmp
    my_out=wtmp
    
    x=my_inp.copy()
    y=my_out.copy()
    
    l2v=layer_2(x,y,c,x.shape[0],c.shape[0],1,sp2)
    l2v.f_mq()
    l2v.ls_solve()
    l2v.pred_f_mq()
    predv=l2v.pred
    
    def plot2(x,y,name):
        plt.plot(x,y, 'o', label='')
        plt.plot([-0.65,1.6],[-0.65,1.6] ,'r')
        plt.xlabel('true',fontsize=16)
        plt.ylabel('pred',fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig('rbf_u',format='png', dpi=100)
        plt.show()
    
    plot2(wtmp_u/10000,predu,'rbf-u-%s'%flist[ii])
    plot2(my_out,predv,'rbf-v-%s'%flist[ii])

    data_rbf=[l2u.w*10000,l2v.w*10000]
    #weights-2 for apprx. weights-1 as func. of Re.
    with open('./rbfout_2/cavity_weight2_w%s_%s.pkl'%(predu.shape[0],predu.shape[1]), 'wb') as outfile:
        pickle.dump(data_rbf, outfile, pickle.HIGHEST_PROTOCOL)
        
#flist=['Re100','Re200','Re300','Re500','Re1000']
#reno=[100,200,300,500,1000]
#fit_layer2(flist,reno)


def pred_layer2(myreno):
  
    my_out=np.zeros((1,200)) #dummy out for shape initialization
    
    my_inp=np.full(200,myreno)
    my_inp=np.reshape(my_inp,(1,len(my_inp)))
    my_inp=my_inp/1000.
    
    x=my_inp
    y=my_out.copy()
    c=c2   
  
    from mrbf_layers import layer_2
    l3u=layer_2(x,y,c,1,c.shape[0],1,sp2)
    l3u.load_weight2(0)
    l3u.pred_f_mq()
    predu=l3u.pred
    
    from mrbf_layers import layer_2
    l3v=layer_2(x,y,c,1,c.shape[0],1,sp2)
    l3v.load_weight2(1)
    l3v.pred_f_mq()
    predv=l3v.pred

    data_rbf=[predu,predv]
    #weights-2 for apprx. weights-1 as func. of Re. - predicted
    with open('./rbfout_2/cavity_weight2_pd_w%s_%s_Re%s.pkl'%(predu.shape[0],predu.shape[1],myreno), 'wb') as outfile:
        pickle.dump(data_rbf, outfile, pickle.HIGHEST_PROTOCOL)

    return predu

predu=pred_layer2(1200)