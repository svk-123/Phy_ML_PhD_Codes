#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
this code used to interpolate LES stress to
RANS coordinates durectly

R-files is created.

@author: vino
"""

# imports
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import interpolate
import cPickle as pickle


#time 
import time
start_time = time.time()

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""

# note-1
'''
#special treament only for ccx,y,z
#for x[]],y[],z[] to read properly
#ex: when,
#    inlet
#    {
#        type            calculated;
#        value           uniform 0;
#        
#    }
#make it, as
#    inlet
#    {
#        type            calculated;
#        value           uniform 0;
#2401
#(
#0
#)
#;
#    }
#    ...
'''

# line no starts with zero in python
bc=['internalField','foil','inlet','outlet','sides']
nbc=[43549,407,204,203,0]

l_bc=np.zeros(len(bc))
ist=np.zeros(len(bc))
iend=np.zeros(len(bc))

path_rans='./rans/rans_naca_0012_aoa_6/330'

path_les=path='./les/les_naca_0012_aoa_6/10'

#load RANS coords
x=[]
with open(path_rans + '/ccx', 'r') as infile:
    data0=infile.readlines()
    
    for i in range(len(bc)):
        line_no=0
        for line in data0:
            if bc[i] in line:
                l_bc[i]=line_no

                for tmp in range(line_no,line_no+5):
                    if str(nbc[i]) in data0[tmp]:
                        ist[i]=tmp+2
                        iend[i]=tmp+2+nbc[i]
                        
                        if ')' in data0[tmp+3]:
                            iend[i]=tmp+2
            
            line_no=line_no+1
    for i in range(len(bc)):
            if(ist[i]==iend[i]):
                for k in range(nbc[i]):
                    x.append(float(line))            
            else:
                for line in data0[int(ist[i]):int(iend[i])]:
                    x.append(float(line)) 
    if (len(x)==sum(nbc)):
        print "X coordinate length correct"
    else:
        print "X length not correct"
x = np.array(map(float, x))
        
       
y=[]
with open(path_rans + '/ccy', 'r') as infile:
    data0=infile.readlines()
    
    for i in range(len(bc)):
        line_no=0
        for line in data0:
            if bc[i] in line:
                l_bc[i]=line_no

                for tmp in range(line_no,line_no+5):
                    if str(nbc[i]) in data0[tmp]:
                        ist[i]=tmp+2
                        iend[i]=tmp+2+nbc[i]
            
            line_no=line_no+1
    for i in range(len(bc)):
            if(ist[i]==iend[i]):
                for k in range(nbc[i]):
                    y.append(float(line))            
            else:
                for line in data0[int(ist[i]):int(iend[i])]:
                    y.append(float(line)) 
    if (len(y)==sum(nbc)):
        print "Y coordinate length correct"
    else:
        print "Y length not correct"
y = np.array(map(float, y))

z=[]
with open(path_rans + '/ccz', 'r') as infile:
    data0=infile.readlines()
    
    for i in range(len(bc)):
        line_no=0
        for line in data0:
            if bc[i] in line:
                l_bc[i]=line_no

                for tmp in range(line_no,line_no+5):
                    if str(nbc[i]) in data0[tmp]:
                        ist[i]=tmp+2
                        iend[i]=tmp+2+nbc[i]
            
            line_no=line_no+1
    for i in range(len(bc)):
            if(ist[i]==iend[i]):
                for k in range(nbc[i]):
                    z.append(float(line))            
            else:
                for line in data0[int(ist[i]):int(iend[i])]:
                    z.append(float(line)) 
    if (len(z)==sum(nbc)):
        print "Z coordinate length correct"
    else:
        print "Z length not correct"
z = np.array(map(float, z))


#load reynols stress
print 'reading R data'
rxx=[]
rxy=[]
rxz=[]
ryy=[]
ryz=[]
rzz=[]

with open(path_rans +'/turbulenceProperties:R', 'r') as infile:
    data0=infile.readlines()
    
    for i in range(len(bc)):
        line_no=0
        for line in data0:
            if bc[i] in line:
                l_bcR[i]=line_no

                for tmp in range(line_no,line_no+5):
                    if str(nbc[i]) in data0[tmp]:
                        istR[i]=tmp+2
                        iendR[i]=tmp+2+nbc[i]
                        
                        if ')' in data0[tmp+3]:
                            if not ')' in data0[tmp+4]:
                                iendR[i]=tmp+2
                                
            line_no=line_no+1                 
        if (int(istR[i]) != int(iendR[i])):
            for line in data0[int(istR[i]):int(iendR[i])]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c,d,e,f = (item.strip() for item in line.split(' ', 6))
                rxx.append(a), rxy.append(b), rxz.append(c),ryy.append(d), ryz.append(e), rzz.append(f)
        else:
            print ("%s - bc has no written value"%bc[i])
            adj=nbc[i]
    if(len(rxx) != (sum(nbc)-adj)):
        print"Error- Length not equal"                            
  
rxx = np.array(map(float, rxx))
rxy = np.array(map(float, rxy))
rxz = np.array(map(float, rxz))
ryy = np.array(map(float, ryy))
ryz = np.array(map(float, ryz))
rzz = np.array(map(float, rzz))





#read LES data
xL=[]
with open(path_les  + '/ccx', 'r') as infile:
    data1=infile.readlines()
    npt=int(data1[20])
    for line in data1[22:22+npt]:
        xL.append(line)
xL = np.array(map(float, xL))

yL=[]
with open(path_les  + '/ccy', 'r') as infile:
    data1=infile.readlines()
    npt=int(data1[20])
    for line in data1[22:22+npt]:
        yL.append(line)
yL = np.array(map(float, yL))

zL=[]
with open(path_les  + '/ccz', 'r') as infile:
    data1=infile.readlines()
    npt=int(data1[20])
    for line in data1[22:22+npt]:
        zL.append(line)
zL = np.array(map(float, zL))

# load velocity
rxxL=[]
rxyL=[]
rxzL=[]
ryyL=[]
ryzL=[]
rzzL=[]

with open(path_les + '/turbulenceProperties:R', 'r') as infile:
    data1=infile.readlines()
    npt=int(data1[20])
    for line in data1[22:22+npt]:
        line=line.replace("(","")
        line=line.replace(")","")        
        a, b, c,d,e,f = (item.strip() for item in line.split(' ', 6))
        rxxL.append(a), rxyL.append(b), rxzL.append(c),ryyL.append(d), ryzL.append(e), rzzL.append(f)
rxxL = np.array(map(float, rxxL))
rxyL = np.array(map(float, rxyL))
rxzL = np.array(map(float, rxzL))
ryyL = np.array(map(float, ryyL))
ryzL = np.array(map(float, ryzL))
rzzL = np.array(map(float, rzzL))


# calculations: no change required
print 'interpolation fn fit...'

#LinearNDinterpolator
pD=np.asarray([xL, yL]).transpose()

fuuD=interpolate.LinearNDInterpolator(pD, rxxL)
fuvD=interpolate.LinearNDInterpolator(pD, rxyL)
fuwD=interpolate.LinearNDInterpolator(pD, rxzL)
fvvD=interpolate.LinearNDInterpolator(pD, ryyL)
fvwD=interpolate.LinearNDInterpolator(pD, ryzL)
fwwD=interpolate.LinearNDInterpolator(pD, rzzL)



#interpolate LES to rans co-ordinates
print 'interpolating...'
for i in range(len(x)):
    rxx[i]=fuuD(x[i],y[i])
    rxy[i]=fuvD(x[i],y[i])
    rxz[i]=fuwD(x[i],y[i])
    ryy[i]=fvvD(x[i],y[i])
    ryz[i]=fvwD(x[i],y[i])
    rzz[i]=fwwD(x[i],y[i])
 
'''
plotD(Z,Y,uvD,20,'DNS')
plot(zs,ys,rxys,20,'rbf')   

plotD(Z,Y,uwD,20,'DNS')
plot(zs,ys,rxzs,20,'rbf')   

plotD(Z,Y,vvD,20,'DNS')
plot(zs,ys,ryys,20,'rbf')   

plotD(Z,Y,vwD,20,'DNS')
plot(zs,ys,ryzs,20,'rbf')   

plotD(Z,Y,wwD,20,'DNS')
plot(zs,ys,rzzs,20,'rbf')    
'''

print 'writing..'
fp= open("./RANS_rans_cr","w+")

for i in range(int(ist[0])):
    fp.write("%s"%(data0[i]))
for i in range(nbc[0]):
    fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i]))
for i in range(int(iend[0]),int(ist[1])):
    fp.write("%s"%(data0[i]))    
for i in range(nbc[0],nbc[0]+nbc[1]):
    fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i])) 
for i in range(int(iend[1]),int(ist[2])):
    fp.write("%s"%(data0[i])) 
for i in range(nbc[0]+nbc[1],nbc[0]+nbc[1]+nbc[2]):
    fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i]))     
for i in range(int(iend[2]),int(ist[3])):
    fp.write("%s"%(data0[i])) 
for i in range(nbc[0]+nbc[1]+nbc[2],nbc[0]+nbc[1]+nbc[2]+nbc[3]):
    fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i]))  
for i in range(int(iend[3]),len(data0)):
    fp.write("%s"%(data0[i]))
    
fp.close() 

print 'DONE'
print("--- %s seconds ---" % (time.time() - start_time))
