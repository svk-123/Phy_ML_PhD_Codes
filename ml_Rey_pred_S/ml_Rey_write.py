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
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate

#time 
import time
start_time = time.time()

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""
# line no starts with zero in python
def write_R_ml(t11,t12,t13,t22,t23,t33):
    # rans coordinate
    bc=['internalField','inlet','outlet','wall']
    nbc=[477799,2401,2401,39004]
    l_bc=np.zeros(len(bc))
    ist=np.zeros(len(bc))
    iend=np.zeros(len(bc))
    
    x=[]
    with open('./rans_data/Re3500/ccx', 'r') as infile:
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
    x = np.array(map(float, x))
            
           
    y=[]
    with open('./rans_data/Re3500/ccy', 'r') as infile:
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
    y = np.array(map(float, y))
    
    z=[]
    with open('./rans_data/Re3500/ccz', 'r') as infile:
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
    z = np.array(map(float, z))
    
    
    #load reynols stress
    print 'reading R-file'
    rxx=[]
    rxy=[]
    rxz=[]
    ryy=[]
    ryz=[]
    rzz=[]
    with open('./rans_data/Re3500/turbulenceProperties:R', 'r') as infile:
        data0=infile.readlines()
        line_no=0
        for line in data0:
            if 'internalField' in line:
                l_intF=line_no
            if 'inlet' in line:
                l_inlet=line_no
            if 'outlet' in line:
                l_outlet=line_no
            if 'wall' in line:
                l_wall=line_no
            line_no=line_no+1           
            
        nintF=int(data0[l_intF+1])
        ninlet=int(data0[l_inlet+4])
        nwall=int(data0[l_wall+4]) 
        noutlet=0  
        
        #internal field
        intFst=l_intF+1+2
        intFend=intFst+nintF
        
        for line in data0[intFst:intFend]:
            line=line.replace("(","")
            line=line.replace(")","")        
            a, b, c,d,e,f = (item.strip() for item in line.split(' ', 6))
            rxx.append(a), rxy.append(b), rxz.append(c),ryy.append(d), ryz.append(e), rzz.append(f)
        
        if(len(rxx) != nintF):
            print"Error- Length not equal"
                
        #inlet
        inletst=l_inlet+4+2
        inletend=inletst+ninlet
        
        for line in data0[inletst:inletend]:
            line=line.replace("(","")
            line=line.replace(")","")        
            a, b, c,d,e,f = (item.strip() for item in line.split(' ', 6))
            rxx.append(a), rxy.append(b), rxz.append(c),ryy.append(d), ryz.append(e), rzz.append(f)
            
        if(len(rxx) != nintF+ninlet):
            print"Error- Length not equal"
            
            
        #wall
        wallst=l_wall+4+2
        wallend=wallst+nwall
        
        for line in data0[wallst:wallend]:
            line=line.replace("(","")
            line=line.replace(")","")        
            a, b, c,d,e,f = (item.strip() for item in line.split(' ', 6))
            rxx.append(a), rxy.append(b), rxz.append(c),ryy.append(d), ryz.append(e), rzz.append(f)
            
        if(len(rxx) != nintF+ninlet+nwall):
            print"Error- Length not equal"   
        
       
    rxx = np.zeros((len(z)))
    rxy = np.zeros((len(z)))
    rxz = np.zeros((len(z)))
    ryy = np.zeros((len(z)))
    ryz = np.zeros((len(z)))
    rzz = np.zeros((len(z)))
    
    rxx[0:2401*199]=np.tile(t11,199)
    rxy[0:2401*199]=np.tile(t12,199)
    rxz[0:2401*199]=np.tile(t13,199)
    ryy[0:2401*199]=np.tile(t22,199)
    ryz[0:2401*199]=np.tile(t23,199)
    rzz[0:2401*199]=np.tile(t33,199)
    
    print 'writing..'
    fp= open("RANS_ml_cr","w+")
    
    for i in range(intFst):
        fp.write("%s"%(data0[i]))
    for i in range(nbc[0]):
        fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i]))
    for i in range(intFend,inletst):
        fp.write("%s"%(data0[i]))    
    for i in range(nbc[0],nbc[0]+nbc[1]):
        fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i])) 
    for i in range(inletend,wallst):
        fp.write("%s"%(data0[i])) 
    for i in range(nbc[0]+nbc[1]+nbc[2],nbc[0]+nbc[1]+nbc[2]+nbc[3]):
        fp.write("(%.12f %.12f %.12f %.12f %.12f %.12f)\n" %(rxx[i],rxy[i],rxz[i],ryy[i],ryz[i],rzz[i]))     
    for i in range(wallend,len(data0)):
        fp.write("%s"%(data0[i]))      
    
    fp.close() 
    
    print 'DONE'
