#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:18:20 2017

@author: vino
"""
import numpy
from matplotlib import pyplot, cm
from numpy import linalg as LA
import math
nx=51
dx=1./(nx-1)
x = numpy.linspace(0, 1, nx)
T=numpy.zeros((nx))
Tb=numpy.zeros((nx))
ep=numpy.zeros((nx))
ep1=numpy.zeros((nx))
R=numpy.zeros((nx))
Tfinal=numpy.zeros((5,nx))
Tin=[5,10,15] # Train1
#Tin=[3,12,14,18,22,28,38,55,60,70] # test1

h=0.5
dt=1e-5

for k in range(3):
    
    T=numpy.zeros((nx))
    Tb=numpy.zeros((nx))
    
    Tinf=Tin[k]
    for it in range(10000):
        for i in range(1,nx-1):
            
            Tt=T.copy()
            
            ep[i]= ( 1 + 5*math.sin( (3*3.14/200)*Tt[i] ) + math.exp(0.02*Tt[i]) )*1e-4
            #ep[i]= 5e-4
                    
            R[i]=((Tt[i-1]-2*Tt[i]+Tt[i+1])/dx**2) + ep[i]*(Tinf**4-Tt[i]**4) #+ h*(Tinf-Tt[i])
            
            T[i]=Tt[i]+dt*R[i]
            
        rss=   LA.norm(T-Tt) / LA.norm(Tt)        
        print "Iter-1= %d\t rss=%.6f\t k=%d\t \n" %(it,rss,k)
        
    for it in range(10000):
        for i in range(1,nx-1):
            
            Tt=Tb.copy()
    
            ep1[i]= 5e-4
            #ep[i]= ( 1 + 5*math.sin( (3*3.14/200)*Tt[i] ) + math.exp(0.02*Tt[i]) )*1e-4
            
            R[i]=((Tt[i-1]-2*Tt[i]+Tt[i+1])/dx**2) + ep1[i]*(Tinf**4-Tt[i]**4) 
            
            Tb[i]=Tt[i]+dt*R[i]   
            
        rss=   LA.norm(Tb-Tt) / LA.norm(Tt)        
        print "Iter-2= %d\t rss=%.6f\t k=%d\t \n" %(it,rss,k)
            
    Tfinal[0,:]=x
    Tfinal[1,:]=Tinf
    Tfinal[2,:]=Tb
    Tfinal[3,:]=T      
    Tfinal[4,:]=ep          
          
    numpy.savetxt('./1d_train1/T%d'%k, Tfinal.transpose(), delimiter=',')
        
    pyplot.figure(figsize=(6, 5), dpi=100)
    l1,=pyplot.plot(x,T,label='True')
    l2,=pyplot.plot(x,Tb,label='Base')
    pyplot.legend()
    pyplot.title('Tinf=%d'%Tinf)
    pyplot.xlabel('X ')
    pyplot.ylabel('T ')
    #pyplot.savefig('./1d_test1/T%d.png'%k, format='png', dpi=100)
    pyplot.show()
