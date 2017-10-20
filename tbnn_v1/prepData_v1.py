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

def get_dns_duct(name,x,y,z):
        
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

def get_rans_duct(path_r):
    
    print 'run...get_rans_duct...'    
    data = np.loadtxt(path_r, skiprows=1)
        
    x,y,z=data[:,0],data[:,1],data[:,2]
    u,v,w=data[:,3],data[:,4],data[:,5]
    k,ep,nut,p=data[:,6],data[:,7],data[:,8],data[:,9]
    
    ux,uy,uz=data[:,10],data[:,11],data[:,12]
    vx,vy,vz=data[:,13],data[:,14],data[:,15]
    wx,wy,wz=data[:,16],data[:,17],data[:,18]
    
    #load when required
    #rxx,rxy,rxz=data[19],data[20],data[21]
    #ryx,ryy,ryz=data[22],data[23],data[24]
    #rzx,rzy,rzz=data[25],data[26],data[27]
        
    grad_u=[ux,uy,uz,vx,vy,vz,wx,wy,wz]
    grad_u=np.asarray(grad_u)
    
    print 'done...get_rans_duct...'      
    return (x,y,z,k,ep,grad_u.transpose())
  
def write_file_duct(Re,path_r,fname,full=True):
    

    flist=Re
    
    #variavles
    xT=[]
    yT=[]
    zT=[]
    kT=[]
    epT=[]
    grad_uT=[]
    uuT=[]
    
    

    x,y,z,k,ep,grad_u=get_rans_duct(path_r)
    uu=get_dns_duct(Re,x,y,z)
        
    for j in range(len(z)):
        if full:
            xT.append(x[j])
            yT.append(y[j])
            zT.append(z[j])
            kT.append(k[j])
            epT.append(ep[j])
            grad_uT.append(grad_u[j])
            uuT.append(uu[j])
            
        else:    
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
    



def get_dns_cbfs(path_d,x,y,z,case):
    
    print 'run...get_dns_cbfs...' 
     
    if (case=='hill'):
        dataframe = pd.read_csv(path_d,sep='\s+',header=None, skiprows=20)
        dataset = dataframe.values
        data=np.asarray(dataset)
        
        """VARIABLES = 0-x,1-y,2-p,3-u/Ub,4-v/Ub,5-w/Ub,6-nu_t/nu,7-uu/Ub^2,8-vv/Ub^2,9-ww/Ub^2,10-uv/Ub^2.
                       11-uw/Ub^2,12-vw/Ub^2,13-k/Ub^2"""
                      
        xD,yD,p,u,v,w,nu,uu,vv,ww,uv,uw,vw,k = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],\
                                             data[:,6],data[:,7],data[:,8],data[:,9],data[:,10],data[:,11],data[:,12],data[:,13]
                                             
    if (case=='cbfs'):
        dataframe = pd.read_csv(path_d,sep='\s+',header=None, skiprows=19)
        dataset = dataframe.values
        data=np.asarray(dataset)
        
        """VARIABLES = 0-x,1-y,2-p,3-u/Ub,4-v/Ub,5-w/Ub,6-uu/Ub^2,7-vv/Ub^2,8-ww/Ub^2,9-uv/Ub^2.
                       10-uw/Ub^2,11-vw/Ub^2,12-k/Ub^2"""
                      
        xD,yD,p,u,v,w,uu,vv,ww,uv,uw,vw,k = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],\
                                             data[:,6],data[:,7],data[:,8],data[:,9],data[:,10],data[:,11],data[:,12]                                        
    if (case=='wavywall'):
        dataframe = pd.read_csv(path_d,sep='\s+',header=None, skiprows=1)
        dataset = dataframe.values
        data=np.asarray(dataset)
        
        """1-x/H  2-z/H  3-u  4-v  5-w  6-p  7-uu  8-vv  9-ww 10-'-uw' 11-pp"""
        #here zD taken ad yD              
        xD,yD,u,v,w,p,uu,vv,ww,uw,pp = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],\
                                             data[:,6],data[:,7],data[:,8],data[:,9],data[:,10]  
        uw=-uw
        y=z
                             
    #interpolate DNS to rans co-ordinates
    
    UUDi=np.zeros((len(x),9))
    
    #LinearNDinterpolator
    pD=np.asarray([xD,yD]).transpose()
    
    print 'interpolation-1...'      
    fuuD=interpolate.LinearNDInterpolator(pD, uu)
    for i in range(len(x)):
        UUDi[i,0]=fuuD(x[i],y[i])
    del fuuD

    
    '''print 'interpolation-2...'    
    fuvD=interpolate.LinearNDInterpolator(pD, uv)
    for i in range(len(x)):
        UUDi[i,1]=fuvD(x[i],y[i])
    del fuvD'''

    
    print 'interpolation-3...'
    fuwD=interpolate.LinearNDInterpolator(pD, uw)
    for i in range(len(x)):    
        UUDi[i,2]=fuwD(x[i],y[i])
    del fuwD    

        
    print 'interpolation-4...'
    fvvD=interpolate.LinearNDInterpolator(pD, vv)
    for i in range(len(x)):
        UUDi[i,4]=fvvD(x[i],y[i])
    del fvvD    

        
    '''print 'interpolation-5...'
    fvwD=interpolate.LinearNDInterpolator(pD, vw)
    for i in range(len(x)):
        UUDi[i,5]=fvwD(x[i],y[i])
    del fvwD'''

        
    print 'interpolation-6...'      
    fwwD=interpolate.LinearNDInterpolator(pD, ww)
    for i in range(len(x)):
        UUDi[i,8]=fwwD(x[i],y[i])
    del fwwD

    print 'interpolation-done...' 
    #--------comment END---------------#
    
    print 'done...get_dns_cbfs...' 
    UUDi[:,3]=UUDi[:,1]
    UUDi[:,6]=UUDi[:,2]
    UUDi[:,7]=UUDi[:,5]
    
    if (np.isnan(UUDi).any()==True):
        inan=[]
        for j in range(len(UUDi[:,0])):
            if (np.isnan(UUDi[j,0])==True):
                inan.append(j)
        
        print 'writing..'
        fp= open("wavywall_inan.txt","w+")
      
        for k in range(len(inan)):
            fp.write("%i\n"%inan[k])
            
        fp.close()
    
    print ('NaN---%s'%np.isnan(UUDi).any())
    return UUDi
 
    
def get_rans_cbfs(path_r):
    
    print 'run...get_rans_cbfs...'    
    data = np.loadtxt(path_r, skiprows=1)
        
    x,y,z=data[:,0],data[:,1],data[:,2]
    u,v,w=data[:,3],data[:,4],data[:,5]
    k,ep,nut,p=data[:,6],data[:,7],data[:,8],data[:,9]
    
    ux,uy,uz=data[:,10],data[:,11],data[:,12]
    vx,vy,vz=data[:,13],data[:,14],data[:,15]
    wx,wy,wz=data[:,16],data[:,17],data[:,18]
    
    #load when required
    #rxx,rxy,rxz=data[19],data[20],data[21]
    #ryx,ryy,ryz=data[22],data[23],data[24]
    #rzx,rzy,rzz=data[25],data[26],data[27]
        
    grad_u=[ux,uy,uz,vx,vy,vz,wx,wy,wz]
    grad_u=np.asarray(grad_u)
    
    print 'done...get_rans_cbfs...'      
    return (x,y,z,k,ep,grad_u.transpose())

    
def write_file_cbfs(path_r,path_d,fname,case,full=True):
    print 'run...write_file_cbfs...'      


    #variavles
    xT=[]
    yT=[]
    zT=[]
    kT=[]
    epT=[]
    grad_uT=[]
    uuT=[]
    
    
    x,y,z,k,ep,grad_u=get_rans_cbfs(path_r)
    uu=get_dns_cbfs(path_d,x,y,z,case)
        
    for j in range(len(x)):
        if full:
            xT.append(x[j])
            yT.append(y[j])
            zT.append(z[j])
            kT.append(k[j])
            epT.append(ep[j])
            grad_uT.append(grad_u[j])
            uuT.append(uu[j])
            
        else:    
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
def plotust(x,y,z,nc,name):
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


def plotst(x,y,z,nc,name):
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

    
    
    
    
    
    
    
    
    
    
    