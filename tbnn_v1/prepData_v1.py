#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:22:23 2017

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

def get_dns(name,x,y,z):
        
    #read DNS data
    dataframe = pd.read_csv('../dns_data/duct/z_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    zD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/y_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    yD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/um_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    uD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/vm_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    vD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/wm_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    wD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/uu_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    uuD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/uv_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    uvD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/uw_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    uwD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/vv_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    vvD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/vw_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    vwD=np.asarray(dataset)
    
    dataframe = pd.read_csv('../dns_data/duct/ww_%s.txt'%name, sep=',', header=None, skiprows=0)
    dataset = dataframe.values
    wwD=np.asarray(dataset)
    

    #----- commment St-----------------#
    # calculations: no change required
    Z, Y = np.meshgrid(zD, yD, copy=False) 
    pD=np.asarray([Z.flatten(), Y.flatten()]).transpose()

    fuuD=interpolate.LinearNDInterpolator(pD, uuD.flatten())
    fuvD=interpolate.LinearNDInterpolator(pD, uvD.flatten())
    fuwD=interpolate.LinearNDInterpolator(pD, uwD.flatten())
    fvvD=interpolate.LinearNDInterpolator(pD, vvD.flatten())
    fvwD=interpolate.LinearNDInterpolator(pD, vwD.flatten())
    fwwD=interpolate.LinearNDInterpolator(pD, wwD.flatten())
    
    #from calc_rans import x,y,z
    #interpolate DNS to rans co-ordinates
    UUDi=np.zeros((len(z),6))
    for i in range(len(z)):
        UUDi[i,0]=fuuD(z[i],y[i])
        UUDi[i,1]=fuvD(z[i],y[i])
        UUDi[i,2]=fuwD(z[i],y[i])
        UUDi[i,3]=fvvD(z[i],y[i])
        UUDi[i,4]=fvwD(z[i],y[i])
        UUDi[i,5]=fwwD(z[i],y[i])
    #--------comment END---------------#

    
    '''with open('./data_out/UUDi_%s.pkl'%name, 'rb') as infile4:
        result4 = pickle.load(infile4)
    UUDi=result4[0]'''
    
    UUDi9=[UUDi[:,0],UUDi[:,1],UUDi[:,2],UUDi[:,1],UUDi[:,3],UUDi[:,4],UUDi[:,2],UUDi[:,4],UUDi[:,5]]
    UUDi9=np.asarray(UUDi9)
    return (UUDi9.transpose())

def get_rans(name,path):
    
    # read RANS data
    # ref:[x,y,z,u,v,w,k,ep,nut]
    with open(path+'duct_rans_data1_%s_r1.pkl'%name, 'rb') as infile1:
        result1 = pickle.load(infile1)
        
    #ref:data2 = [ux,uy,uz,vx,vy,vz,wx,wy,wz]
    with open(path+'duct_rans_data2_%s_r1.pkl'%name, 'rb') as infile2:
        result2 = pickle.load(infile2)
    
    #ref:data3 = [rxx,rxy,rxz,ryy,ryz,rzz]
    with open(path+'duct_rans_data3_%s_r1.pkl'%name, 'rb') as infile3:
        result3 = pickle.load(infile3)
        
    x,y,z=result1[0],result1[1],result1[2]
    u,v,w=result1[3],result1[4],result1[5]
    k,ep,mut=result1[6],result1[7],result1[8]
    
    ux,uy,uz=result2[0],result2[1],result2[2]
    vx,vy,vz=result2[3],result2[4],result2[5]
    wx,wy,wz=result2[6],result2[7],result2[8]
    
    rxx,rxy,rxz=result3[0],result3[1],result3[2]
    ryy,ryz,rzz=result3[3],result3[4],result3[5]
    
    grad_u=[ux,uy,uz,vx,vy,vz,wx,wy,wz]
    grad_u=np.asarray(grad_u)
    return (x,y,z,k,ep,grad_u.transpose())
  
def write_file(mylist,path,fname):
    

    flist=mylist
    
    #variavles
    xT=[]
    yT=[]
    zT=[]
    kT=[]
    epT=[]
    grad_uT=[]
    uuT=[]
    
    
    for ii in range(len(flist)):
        x,y,z,k,ep,grad_u=get_rans(flist[ii],path)
        uu=get_dns(flist[ii],x,y,z)
        
        for j in range(len(z)):
            if(z[j]<=0.15):
             
                xT.append(x[j])
                yT.append(y[j])
                zT.append(z[j])
                kT.append(k[j])
                epT.append(ep[j])
                grad_uT.append(grad_u[j])
                uuT.append(uu[j])    
    
    xT=np.asarray(xT)
    yT=np.asarray(yT)
    zT=np.asarray(zT)
    kT=np.asarray(kT)
    epT=np.asarray(epT)
    grad_uT=np.asarray(grad_uT)
    uuT=np.asarray(uuT)
    

  
    l=len(xT)     
    
    print 'writing..'
    fp= open("./datafile/%s.txt"%fname,"w+")
    
    fp.write('k, ep, ux, uy, uz, vx, vy, vz, wx, wy, wz, uu, uv, uw, vu, vv,vw,wu, wv, ww, x, y, z\n')
    
    for i in range(l):
        fp.write("%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"\
                 %(kT[i],epT[i],grad_uT[i,0],grad_uT[i,1],grad_uT[i,2],grad_uT[i,3],grad_uT[i,4],grad_uT[i,5],grad_uT[i,6],grad_uT[i,7],grad_uT[i,8],\
                   uuT[i,0],uuT[i,1],uuT[i,2],uuT[i,3],uuT[i,4],uuT[i,5],uuT[i,6],uuT[i,7],uuT[i,8],xT[i],yT[i],zT[i]))        
     
    fp.close() 
    


def load_data(fname):
    """
    Loads in channel flow data
    :return:
    """

    # Load in data from channel.txt
    data = np.loadtxt('./datafile/%s.txt'%fname, skiprows=1)
    k = data[:, 0]
    eps = data[:, 1]
    grad_u_flat = data[:, 2:11]
    stresses_flat = data[:, 11:20]
    coord=data[:,20:23]
    # Reshape grad_u and stresses to num_points X 3 X 3 arrays
    num_points = data.shape[0]
    grad_u = np.zeros((num_points, 3, 3))
    stresses = np.zeros((num_points, 3, 3))
    for i in xrange(3):
        for j in xrange(3):
            grad_u[:, i, j] = grad_u_flat[:, i*3+j]
            stresses[:, i, j] = stresses_flat[:, i*3+j]
    return k, eps, grad_u, stresses,coord    

#plot
def plot(x,y,z,nc,name):
    pyplot.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = pyplot.tricontourf(x, y, z,nc,cmap=cm.jet)
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    pyplot.colorbar()
    pyplot.title(name)
    pyplot.xlabel('Z ')
    pyplot.ylabel('Y ')
    #pyplot.savefig(name, format='png', dpi=100)
    pyplot.show()


def plotD(x,y,z,nc,name):
    pyplot.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = pyplot.contourf(x, y, z,nc,cmap=cm.jet)
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    pyplot.colorbar()
    pyplot.title(name)
    pyplot.xlabel('Z ')
    pyplot.ylabel('Y ')
    #pyplot.savefig(name, format='png', dpi=100)
    pyplot.show()
    
def test_file():
    print "not implemnted"    

    
    
    
    
    
    
    
    
    
    
    