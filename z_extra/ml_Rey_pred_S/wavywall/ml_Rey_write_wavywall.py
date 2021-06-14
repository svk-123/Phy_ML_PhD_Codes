#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:31:27 2017

@author: vino
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 19:35:45 2017

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
import dill
import klepto

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

# path of Rey file to read
path='/home/vino/OpenFOAM/run/mycase/zpyPost_gen/rans_data/wavywall/'
name='Re6760'

def write_R_ml(t11,t12,t13,t22,t23,t33,xR,yR,zR):
        
    # line no starts with zero in python
    # for 200h duct
    #bc=['internalField','inlet','mywall','outlet']
    #nbc=[2085119,3481,141364,3481]
    
    # for cbfs
    bc=['internalField','inlet','mywall','outlet','side']
    nbc=[1950399,0,39402,0,0]
    
    l_bc=np.zeros(len(bc))
    ist=np.zeros(len(bc))
    iend=np.zeros(len(bc))
    
    x=[]
    with open(path+'%s/ccx'%name, 'r') as infile:
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
    with open(path+'%s/ccy'%name, 'r') as infile:
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
    with open(path+'%s/ccz'%name, 'r') as infile:
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
    mytmp=[]
    
    l_bcR=np.zeros(len(bc))
    istR=np.zeros(len(bc))
    iendR=np.zeros(len(bc))
    
    with open(path+'%s/turbulenceProperties:R'%name, 'r') as infile:
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
                    mytmp.append(a)
            else:
                print ("%s - bc has no written value"%bc[i])
                adj=nbc[i]
        if(len(mytmp) != (sum(nbc)-adj-19701-19701)):
            print"Error- Length not equal"   
        else:
            print 'R - Length Okay'
                        

    rxx = np.zeros((len(z)))
    rxy = np.zeros((len(z)))
    rxz = np.zeros((len(z)))
    ryy = np.zeros((len(z)))
    ryz = np.zeros((len(z)))
    rzz = np.zeros((len(z)))
    
    '''
    rxx[0:19701*199]=np.tile(t11,199)
    rxy[0:19701*199]=np.tile(t12,199)
    rxz[0:19701*199]=np.tile(t13,199)
    ryy[0:19701*199]=np.tile(t22,199)
    ryz[0:19701*199]=np.tile(t23,199)
    rzz[0:19701*199]=np.tile(t33,199)
    
    

    l=nbc[0]
    ix=np.arange(l)
    da=np.zeros((l,10))

    da[:,0]=x[:l]
    da[:,1]=y[:l]
    da[:,2]=z[:l]
    da[:,3]=0.0
    da[:,4]=0.0
    da[:,5]=0.0
    da[:,6]=0.0
    da[:,7]=0.0
    da[:,8]=0.0
    da[:,9]=ix
    

    da=da[np.argsort(da[:,2])]
    
  
    da[:,3]=rxx[:l]
    da[:,4]=rxy[:l]
    da[:,5]=rxz[:l]
    da[:,6]=ryy[:l]
    da[:,7]=ryz[:l]
    da[:,8]=rzz[:l]
    '''
    

    #LinearNDinterpolator
    pD=np.asarray([xR,zR]).transpose()
    
    print 'interpolation-1...'      
    frxx=interpolate.LinearNDInterpolator(pD,t11)
    for i in range(nbc[0]):
        rxx[i]=frxx(x[i],y[i])
   

    for i in range(nbc[0]):
        rxy[i]=0
        
        
    print 'interpolation-1...'      
    frxz=interpolate.LinearNDInterpolator(pD,t13)
    for i in range(nbc[0]):
        rxz[i]=frxz(x[i],y[i])
                
    print 'interpolation-1...'      
    fryy=interpolate.LinearNDInterpolator(pD,t22)
    for i in range(nbc[0]):
        ryy[i]=fryy(x[i],y[i])
       

    for i in range(nbc[0]):
        ryz[i]=0
        
    print 'interpolation-1...'      
    frzz=interpolate.LinearNDInterpolator(pD,t33)
    for i in range(nbc[0]):
        rzz[i]=frzz(x[i],y[i])
   
    inan=[]
    for i in range(len(rxx)):
        if (np.isnan(rxx[i])==True):
            inan.append(i)
            rxx[i]=0
        if (np.isnan(rxy[i])==True):
            inan.append(i)
            rxy[i]=0       
        if (np.isnan(rxz[i])==True):
            inan.append(i)
            rxz[i]=0              
        if (np.isnan(ryy[i])==True):
            inan.append(i)
            ryy[i]=0              
        if (np.isnan(ryz[i])==True):
            inan.append(i)
            ryz[i]=0              
        if (np.isnan(rzz[i])==True):
            inan.append(i)
            rzz[i]=0              
            
            
    # only for plotting        
    l=nbc[0]
    ix=np.arange(l)
    da=np.zeros((l,5))

    da[:,0]=x[:l]
    da[:,1]=y[:l]
    da[:,2]=z[:l]
    da[:,3]=rxx[:l]
    da[:,4]=ix
    

    da=da[np.argsort(da[:,2])]
    #da=da[np.argsort(da[:,9])]    
    #------------------

    print 'done'
           

    print 'writing..'
    fp= open("RANS_ml_wavywall","w+")
    
    for i in range(int(istR[0])):
        fp.write("%s"%(data0[i]))
    for i in range(nbc[0]):
        fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i]))
    for i in range(int(iendR[0]),int(istR[2])):
        fp.write("%s"%(data0[i]))    
    for i in range(nbc[0],nbc[0]+nbc[2]):
        fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i])) 
    
    
    #skip outlet bc[3], inlet,side
    #no written value
    for i in range(int(iendR[2]),len(data0)):
        fp.write("%s"%(data0[i])) 
    
    fp.close() 

    print istR
    print iendR
    
    
    def plot(x,y,z,nc,name):
        fig=plt.figure(figsize=(6, 5), dpi=100)
        ax=fig.add_subplot(111)
        #cp = ax.tricontourf(x, y, z,np.linspace(-0.3,0.3,30),extend='both')
        cp = ax.tricontourf(x, y, z,30,extend='both')
        #cp.set_clim(-0.2,0.2)
        #plt.xlim([-1, 0])
        #plt.ylim([-1, 0])
         
        cbar=plt.colorbar(cp)
        plt.title(name)
        plt.xlabel('Z ')
        plt.ylabel('Y ')
        #plt.savefig(name +'.png', format='png', dpi=100)
        plt.show()
        

    plot(xR,yR,t11,20,'t11')   
    l=19701
    
    mylist=[0,100,198]
    for k in mylist:
        a=0+k*l
        b=l+k*l

        plot(da[a:b,0],da[a:b,2],da[a:b,3],20,'name')  
    
    #plot(da[0:19701,0],da[0:19701,1],da[0+100*19701:19701+100*19701,3],20,'rxx') 
    #plt.scatter(da[0:19701,0],da[0:19701,1])
    
    
    
    
    print 'DONE'
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #return (inan,istR,iendR)