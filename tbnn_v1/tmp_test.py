# imports
import os
import glob

import numpy as np
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate
import cPickle as pickle

print 'run...get_dns_cbfs...'  


path_r='../rans_data/hill/hill_Re10595_train_wnan.txt'
path_d='../dns_data/hill/Re10595/hill_train.dat'

case='hill'





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

 
print 'run...get_rans_cbfs...'    
data = np.loadtxt(path_r, skiprows=1)
        
x,y,z=data[:,0],data[:,1],data[:,2]
u,v,w=data[:,3],data[:,4],data[:,5]
k,ep,nut,p=data[:,6],data[:,7],data[:,8],data[:,9]
    
ux,uy,uz=data[:,10],data[:,11],data[:,12]
vx,vy,vz=data[:,13],data[:,14],data[:,15]
wx,wy,wz=data[:,16],data[:,17],data[:,18]


#interpolate DNS to rans co-ordinates

UUDi=np.zeros((len(x),9))

#LinearNDinterpolator
pD=np.asarray([xD,yD]).transpose()

print 'interpolation-1...'  
fuuD=interpolate.LinearNDInterpolator(pD, uu)
for i in range(len(x)):
    UUDi[i,0]=fuuD(x[i],y[i])
del fuuD

inan=[]
for i in range(len(UUDi[:,0])):
    if (np.isnan(UUDi[i,0])==True):
        inan.append(i)
        
'''print 'interpolation-2...'
fuvD=interpolate.LinearNDInterpolator(pD, uv)
for i in range(len(x)):
    UUDi[i,1]=fuvD(x[i],y[i])
del fuvD
print np.isnan(UUDi[:,1]).any()'''

print 'interpolation-3...'
fuwD=interpolate.LinearNDInterpolator(pD, uw)
for i in range(len(x)):
    UUDi[i,2]=fuwD(x[i],y[i])
del fuwD
print np.isnan(UUDi[:,2]).any()

print 'interpolation-4...'
fvvD=interpolate.LinearNDInterpolator(pD, vv)
for i in range(len(x)):
    UUDi[i,4]=fvvD(x[i],y[i])
del fvvD    
print np.isnan(UUDi[:,4]).any()
    
'''print 'interpolation-5...'
fvwD=interpolate.LinearNDInterpolator(pD, vw)
for i in range(len(x)):
    UUDi[i,5]=fvwD(x[i],y[i])
del fvwD  
print np.isnan(UUDi[:,5]).any()'''
    
print 'interpolation-6...'      
fwwD=interpolate.LinearNDInterpolator(pD, ww)
for i in range(len(x)):
    UUDi[i,8]=fwwD(x[i],y[i])
del fwwD
print np.isnan(UUDi[:,8]).any()

for k in range(len(inan)):
    #print UUDi[inan[k],1],UUDi[inan[k],2],UUDi[inan[k],3],UUDi[inan[k],4],UUDi[inan[k],5]
    print x[inan[k]],y[inan[k]]

print 'writing..'
fp= open("hill_inan.txt","w+")
  
for k in range(len(inan)):
    fp.write("%i\n"%inan[k])
         
     
fp.close()


#plot
def plotust(x,y,z,nc,name):
    pyplot.figure(figsize=(9, 5), dpi=100)
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
    
plotust(x,y,u,20,'name')    
    
#pyplot.scatter(x,y)