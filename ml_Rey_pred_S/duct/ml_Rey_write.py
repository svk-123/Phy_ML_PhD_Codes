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
path='/home/vino/OpenFOAM/run/mycase/zpyPost_gen/rans_data/data_r0/'
name='Re3500'

def write_R_ml(t11,t12,t13,t22,t23,t33):
        
    # line no starts with zero in python
    # for 200h duct
    #bc=['internalField','inlet','mywall','outlet']
    #nbc=[2085119,3481,141364,3481]
    
    # for 100h duct
    bc=['internalField','inlet','wall','outlet']
    nbc=[477799,2401,39004,2401]
    
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
        if(len(mytmp) != (sum(nbc)-adj)):
            print"Error- Length not equal"    
                        
    rxx = np.zeros((len(z)))
    rxy = np.zeros((len(z)))
    rxz = np.zeros((len(z)))
    ryy = np.zeros((len(z)))
    ryz = np.zeros((len(z)))
    rzz = np.zeros((len(z)))
    
    '''rxx[0:3481*599]=np.tile(t11,599)
    rxy[0:3481*599]=np.tile(t12,599)
    rxz[0:3481*599]=np.tile(t13,599)
    ryy[0:3481*599]=np.tile(t22,599)
    ryz[0:3481*599]=np.tile(t23,599)
    rzz[0:3481*599]=np.tile(t33,599)'''  
    
  
      
    
    
    rxx[0:2401*199]=np.tile(t11,199)
    rxy[0:2401*199]=np.tile(t12,199)
    rxz[0:2401*199]=np.tile(t13,199)
    ryy[0:2401*199]=np.tile(t22,199)
    ryz[0:2401*199]=np.tile(t23,199)
    rzz[0:2401*199]=np.tile(t33,199)  
    
    print 'writing..'
    fp= open("RANS_DNS_cr_100hk","w+")
    
    for i in range(int(istR[0])):
        fp.write("%s"%(data0[i]))
    for i in range(nbc[0]):
        fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i]))
    for i in range(int(iendR[0]),int(istR[1])):
        fp.write("%s"%(data0[i]))    
    for i in range(nbc[0],nbc[0]+nbc[1]):
        fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i])) 
    for i in range(int(iendR[1]),int(istR[2])):
        fp.write("%s"%(data0[i])) 
    for i in range(nbc[0]+nbc[1],nbc[0]+nbc[1]+nbc[2]):
        fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i]))     
    
    #skip outlet bc[3]
    #no written value
    for i in range(int(iendR[2]),len(data0)):
        fp.write("%s"%(data0[i]))      
    
    fp.close() 
    
    print 'DONE'
    print("--- %s seconds ---" % (time.time() - start_time))