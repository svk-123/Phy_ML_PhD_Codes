#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:51:09 2017

@author: vino
"""
# imports
import os
import glob

import numpy as np
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate
import cPickle as pickle

def calc_rans(k,ep,grad_u):
    
    l=len(grad_u)
    
    '''        
    x,y,z=result1[0],result1[1],result1[2]
    u,v,w=result1[3],result1[4],result1[5]
    k,ep,mut=result1[6],result1[7],result1[8]
    
    ux,uy,uz=result2[0],result2[1],result2[2]
    vx,vy,vz=result2[3],result2[4],result2[5]
    wx,wy,wz=result2[6],result2[7],result2[8]
    
    rxx,rxy,rxz=result3[0],result3[1],result3[2]
    ryy,ryz,rzz=result3[3],result3[4],result3[5]
    '''
    
            
    dU=grad_u.reshape(l,3,3)
    dUT=np.zeros((l,3,3))
    for i in range(l):
        dUT[i,:,:]=dU[i,:,:].transpose()

    S=0.5*(dU+dUT)
    R=0.5*(dU-dUT)    
    
    # Non dim S,R
    for i in range(l,):
        S[i,:,:]=S[i,:,:]*(k[i]/ep[i])
        R[i,:,:]=R[i,:,:]*(k[i]/ep[i])    
        
           
    TrS2  =np.zeros(l)
    TrR2  =np.zeros(l)
    TrS3  =np.zeros(l)
    TrR2S =np.zeros(l)
    TrR2S2=np.zeros(l)
    TrSR2 =np.zeros(l)
    TrS2R2=np.zeros(l)
    
    for i in range(l):   
        TrS2[i]=np.trace(np.matmul(S[i,:,:],S[i,:,:]))
        TrR2[i]=np.trace(np.matmul(R[i,:,:],R[i,:,:]))
        TrS3[i]=np.trace( np.matmul(np.matmul(S[i,:,:],S[i,:,:]),S[i,:,:]) )
        TrR2S[i]=np.trace( np.matmul(np.matmul(R[i,:,:],R[i,:,:]),S[i,:,:]) )    
        TrR2S2[i]=np.trace( np.matmul(np.matmul(R[i,:,:],R[i,:,:]),np.matmul(S[i,:,:],S[i,:,:])) )       
        TrSR2[i]=np.trace( np.matmul(S[i,:,:],np.matmul(R[i,:,:],R[i,:,:])) )     
        TrS2R2[i]=np.trace( np.matmul(np.matmul(S[i,:,:],S[i,:,:]),np.matmul(R[i,:,:],R[i,:,:])) )       
        
    I=np.zeros((l,3,3))   
    for i in range(l):
        I[i,:,:]=np.eye(3)
        
    T1=np.zeros((l,3,3))
    T2=np.zeros((l,3,3))
    T3=np.zeros((l,3,3))
    T4=np.zeros((l,3,3))
    T5=np.zeros((l,3,3))
    T6=np.zeros((l,3,3))
    T7=np.zeros((l,3,3))
    T8=np.zeros((l,3,3))
    T9=np.zeros((l,3,3))
    T10=np.zeros((l,3,3))
    
    for i in range(l):     
        T1[i,:,:]=S[i,:,:]
        T2[i,:,:]=np.matmul(S[i,:,:],R[i,:,:]) - np.matmul(R[i,:,:],S[i,:,:])
        T3[i,:,:]=np.matmul(S[i,:,:],S[i,:,:]) - (1.0/3.0)*I[i,:,:]*TrS2[i]   
        T4[i,:,:]=np.matmul(R[i,:,:],R[i,:,:]) - (1.0/3.0)*I[i,:,:]*TrR2[i]
        T5[i,:,:]=np.matmul(R[i,:,:],np.matmul(S[i,:,:],S[i,:,:])) - np.matmul(np.matmul(S[i,:,:],S[i,:,:]),R[i,:,:])
        T6[i,:,:]=np.matmul(np.matmul(R[i,:,:],R[i,:,:]),S[i,:,:]) + np.matmul(S[i,:,:],np.matmul(R[i,:,:],R[i,:,:]))-(2.0/3.0)*I[i,:,:]*TrSR2[i]
        T7[i,:,:]=np.matmul(np.matmul(np.matmul(R[i,:,:],S[i,:,:]),R[i,:,:]),R[i,:,:]) - np.matmul(np.matmul(np.matmul(R[i,:,:],R[i,:,:]),S[i,:,:]),R[i,:,:])
        T8[i,:,:]=np.matmul(np.matmul(np.matmul(S[i,:,:],R[i,:,:]),S[i,:,:]),S[i,:,:]) - np.matmul(np.matmul(np.matmul(S[i,:,:],S[i,:,:]),R[i,:,:]),S[i,:,:])
        T9[i,:,:]=np.matmul(np.matmul(np.matmul(R[i,:,:],R[i,:,:]),S[i,:,:]),S[i,:,:]) + np.matmul(np.matmul(np.matmul(S[i,:,:],S[i,:,:]),R[i,:,:]),R[i,:,:])\
                  -(2.0/3.0)*I[i,:,:]*TrS2R2[i]
        T10[i,:,:]=np.matmul(np.matmul(np.matmul(np.matmul(R[i,:,:],S[i,:,:]),S[i,:,:]),R[i,:,:]),R[i,:,:]) - np.matmul(np.matmul(np.matmul(np.matmul(R[i,:,:],R[i,:,:]),S[i,:,:]),S[i,:,:]),R[i,:,:])
                 
    L1=TrS2
    L2=TrR2
    L3=TrS3
    L4=TrR2S
    L5=TrR2S2
    
    L=np.zeros((l,5))
    for i in range(l):
        L[i,0]=L1[i]
        L[i,1]=L2[i]
        L[i,2]=L3[i]
        L[i,3]=L4[i]
        L[i,4]=L5[i]
        
    #Components-6 Input for ML
    T1c=np.zeros((l,6))
    T2c=np.zeros((l,6))
    T3c=np.zeros((l,6))
    T4c=np.zeros((l,6))
    T5c=np.zeros((l,6))
    T6c=np.zeros((l,6))
    T7c=np.zeros((l,6))
    T8c=np.zeros((l,6))
    T9c=np.zeros((l,6))
    T10c=np.zeros((l,6))
    
    for i in range(l):
        T1c[i,0]=T1[i,0,0]
        T1c[i,1]=T1[i,0,1]
        T1c[i,2]=T1[i,0,2]
        T1c[i,3]=T1[i,1,1]
        T1c[i,4]=T1[i,1,2]
        T1c[i,5]=T1[i,2,2]
                      
        T2c[i,0]=T2[i,0,0]
        T2c[i,1]=T2[i,0,1]
        T2c[i,2]=T2[i,0,2]
        T2c[i,3]=T2[i,1,1]
        T2c[i,4]=T2[i,1,2]
        T2c[i,5]=T2[i,2,2]
        
        T3c[i,0]=T3[i,0,0]
        T3c[i,1]=T3[i,0,1]
        T3c[i,2]=T3[i,0,2]
        T3c[i,3]=T3[i,1,1]
        T3c[i,4]=T3[i,1,2]
        T3c[i,5]=T3[i,2,2]
        
        T4c[i,0]=T4[i,0,0]
        T4c[i,1]=T4[i,0,1]
        T4c[i,2]=T4[i,0,2]
        T4c[i,3]=T4[i,1,1]
        T4c[i,4]=T4[i,1,2]
        T4c[i,5]=T4[i,2,2]
        
        T5c[i,0]=T5[i,0,0]
        T5c[i,1]=T5[i,0,1]
        T5c[i,2]=T5[i,0,2]
        T5c[i,3]=T5[i,1,1]
        T5c[i,4]=T5[i,1,2]
        T5c[i,5]=T5[i,2,2]
        
        T6c[i,0]=T6[i,0,0]
        T6c[i,1]=T6[i,0,1]
        T6c[i,2]=T6[i,0,2]
        T6c[i,3]=T6[i,1,1]
        T6c[i,4]=T6[i,1,2]
        T6c[i,5]=T6[i,2,2]
        
        T7c[i,0]=T7[i,0,0]
        T7c[i,1]=T7[i,0,1]
        T7c[i,2]=T7[i,0,2]
        T7c[i,3]=T7[i,1,1]
        T7c[i,4]=T7[i,1,2]
        T7c[i,5]=T7[i,2,2]
        
        T8c[i,0]=T8[i,0,0]
        T8c[i,1]=T8[i,0,1]
        T8c[i,2]=T8[i,0,2]
        T8c[i,3]=T8[i,1,1]
        T8c[i,4]=T8[i,1,2]
        T8c[i,5]=T8[i,2,2]
        
        T9c[i,0]=T9[i,0,0]
        T9c[i,1]=T9[i,0,1]
        T9c[i,2]=T9[i,0,2]
        T9c[i,3]=T9[i,1,1]
        T9c[i,4]=T9[i,1,2]
        T9c[i,5]=T9[i,2,2]
        
        T10c[i,0]=T10[i,0,0]
        T10c[i,1]=T10[i,0,1]
        T10c[i,2]=T10[i,0,2]
        T10c[i,3]=T10[i,1,1]
        T10c[i,4]=T10[i,1,2]
        T10c[i,5]=T10[i,2,2]
        
        
        
    Tc=np.zeros((l,10,6))
    for i in range(l):
        Tc[i,0,:]=T1c[i,:]
        Tc[i,1,:]=T2c[i,:]
        Tc[i,2,:]=T3c[i,:]
        Tc[i,3,:]=T4c[i,:]
        Tc[i,4,:]=T5c[i,:]
        Tc[i,5,:]=T6c[i,:]
        Tc[i,6,:]=T7c[i,:]
        Tc[i,7,:]=T8c[i,:]
        Tc[i,8,:]=T9c[i,:]
        Tc[i,9,:]=T10c[i,:]
     
    
    
    # 10 elements of six : for dot product
    T1m=np.zeros((l,10))
    T2m=np.zeros((l,10))
    T3m=np.zeros((l,10))
    T4m=np.zeros((l,10))
    T5m=np.zeros((l,10))
    T6m=np.zeros((l,10))
    
    for i in range(l):
        T1m[i,0]=T1c[i,0]
        T1m[i,1]=T2c[i,0]
        T1m[i,2]=T3c[i,0]
        T1m[i,3]=T4c[i,0]
        T1m[i,4]=T5c[i,0]
        T1m[i,5]=T6c[i,0]
        T1m[i,6]=T7c[i,0]
        T1m[i,7]=T8c[i,0]
        T1m[i,8]=T9c[i,0]
        T1m[i,9]=T10c[i,0]    
        
    for i in range(l):
        T2m[i,0]=T1c[i,1]
        T2m[i,1]=T2c[i,1]
        T2m[i,2]=T3c[i,1]
        T2m[i,3]=T4c[i,1]
        T2m[i,4]=T5c[i,1]
        T2m[i,5]=T6c[i,1]
        T2m[i,6]=T7c[i,1]
        T2m[i,7]=T8c[i,1]
        T2m[i,8]=T9c[i,1]
        T2m[i,9]=T10c[i,1]        
        
    for i in range(l):
        T3m[i,0]=T1c[i,2]
        T3m[i,1]=T2c[i,2]
        T3m[i,2]=T3c[i,2]
        T3m[i,3]=T4c[i,2]
        T3m[i,4]=T5c[i,2]
        T3m[i,5]=T6c[i,2]
        T3m[i,6]=T7c[i,2]
        T3m[i,7]=T8c[i,2]
        T3m[i,8]=T9c[i,2]
        T3m[i,9]=T10c[i,2]      
        
    for i in range(l):
        T4m[i,0]=T1c[i,3]
        T4m[i,1]=T2c[i,3]
        T4m[i,2]=T3c[i,3]
        T4m[i,3]=T4c[i,3]
        T4m[i,4]=T5c[i,3]
        T4m[i,5]=T6c[i,3]
        T4m[i,6]=T7c[i,3]
        T4m[i,7]=T8c[i,3]
        T4m[i,8]=T9c[i,3]
        T4m[i,9]=T10c[i,3]      
        
    for i in range(l):
        T5m[i,0]=T1c[i,4]
        T5m[i,1]=T2c[i,4]
        T5m[i,2]=T3c[i,4]
        T5m[i,3]=T4c[i,4]
        T5m[i,4]=T5c[i,4]
        T5m[i,5]=T6c[i,4]
        T5m[i,6]=T7c[i,4]
        T5m[i,7]=T8c[i,4]
        T5m[i,8]=T9c[i,4]
        T5m[i,9]=T10c[i,4]    
        
    for i in range(l):
        T6m[i,0]=T1c[i,5]
        T6m[i,1]=T2c[i,5]
        T6m[i,2]=T3c[i,5]
        T6m[i,3]=T4c[i,5]
        T6m[i,4]=T5c[i,5]
        T6m[i,5]=T6c[i,5]
        T6m[i,6]=T7c[i,5]
        T6m[i,7]=T8c[i,5]
        T6m[i,8]=T9c[i,5]
        T6m[i,9]=T10c[i,5]
        
    return (L,T1m,T2m,T3m,T4m,T5m,T6m,S,R)
  
    
    
    
    
    
    
    
    
