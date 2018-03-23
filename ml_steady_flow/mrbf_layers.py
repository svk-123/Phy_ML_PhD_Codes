#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

@author: vinoth
"""

import numpy as np
import cPickle as pickle

class layer_1(object):
    
    """Implementation of Layer-1.
    
    For cavity flow prediction problem, this layer takes x,y as input. Rbf is formulated and weight-1 is 
    obtained for a particular Re.
    
    Input: x,y
    Output:u or v
    get weight co-effs: w (passed to layer 2)
    
    Lx - length of input.shape[0]
    Lc - length of centers
    d  - length of input dim 
    sp - shape parameter
       
    """
       
    def __init__(self, x,y,c,Lx,Lc,d,sp):
        self.Lx = Lx
        self.Lc = Lc
        self.d  = d
        self.sp = sp
        self.x = x
        self.y = y
        self.c = c
        self.P = np.zeros((self.Lx,self.Lc))
        self.pred = np.zeros((self.y.shape))
        
    def f_mq(self):
        """ multi quadratic"""
        """f=( (x-xc)^2 + c^2 )^1/2"""
        for i in range(self.Lx):
            for j in range(self.Lc):
                tmp=0
                for l in range(self.d):
                    tmp+=(self.x[i,l]-self.c[j,l])**2
                self.P[i,j]=np.sqrt(tmp+(self.sp**2))
            
    def ls_solve(self):
        """ LS solver """
        self.w,self.res,self.Prank,_=np.linalg.lstsq(self.P,self.y,rcond=None)

    def pred_f_mq(self):
        for i in range(self.Lx):
            tmp1=0
            for j in range(self.Lc):
                tmp2=0
                for l in range(self.d):
                    tmp2+=(self.x[i,l]-self.c[j,l])**2
                tmp2=np.sqrt(tmp2+(self.sp**2))
                tmp1+=self.w[j]*tmp2
            self.pred[i]=tmp1
        

    def load_weight1(self,val):
        """ load weight dueing prediction"""
        pass


class layer_2(object):

    """Implementation of Layer-2
    
    This layer approximates the first layer weight with given input.
    For cavity case, first layer weights are approximated as a func. of Re.
            
    """
        
    def __init__(self, x,y,c,Lx,Lc,d,sp):
        self.Lx = Lx
        self.Lc = Lc
        self.d  = d
        self.sp = sp
        self.x = x
        self.y = y
        self.c = c
        self.P = np.zeros((self.Lx,self.c.shape[0]))
        self.pred = np.zeros((self.y.shape[0],self.y.shape[1]))


    def f_mq(self):
        """ multi quadratic"""
        """f=( (x-xc)^2 + c^2 )^1/2"""
        for i in range(self.Lx):
            for j in range(self.Lc):
                tmp=0
                for l in range(self.d):
                    tmp+=(self.x[i,l]-self.c[j,l])**2
                self.P[i,j]=np.sqrt(tmp+(self.sp**2))


    def ls_solve(self):
        """ LS solver """
        self.w,self.res,self.Prank,_=np.linalg.lstsq(self.P,self.y,rcond=None)


    def pred_f_mq(self):
        for i in range(self.Lx):
            for m in range(self.y.shape[1]):
                tmp1=0
                for j in range(self.Lc):
                    tmp2=0
                    for l in range(self.d):
                        tmp2+=(self.x[i,m]-self.c[j,l])**2
                    tmp2=np.sqrt(tmp2+(self.sp**2))
                    tmp1+=self.w[j,m]*tmp2
                self.pred[i,m]=tmp1

    
    def load_weight2(self,val):
        """ load weight 2 during prediction"""
        """ val=1 for u-eight, 2 for v-weigth"""
        
        with open('./rbfout_2/cavity_weight2_w5_200.pkl', 'rb') as infile:
            result = pickle.load(infile)
            self.w=result[val]