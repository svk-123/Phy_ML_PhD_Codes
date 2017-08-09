# imports
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import interpolate
import cPickle as pickle

#load reynols stress
#storing stripped Reynolds stress
#Not required Now. But can be used later

def write_R(rxx,rxy,rxz,ryy,ryz,rzz):
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
            #rxx.append(a), rxy.append(b), rxz.append(c),ryy.append(d), ryz.append(e), rzz.append(f)
        
        if(len(rxx) != nintF):
            print"Error- Length not equal"
                
        #inlet
        inletst=l_inlet+4+2
        inletend=inletst+ninlet
        
        for line in data0[inletst:inletend]:
            line=line.replace("(","")
            line=line.replace(")","")        
            a, b, c,d,e,f = (item.strip() for item in line.split(' ', 6))
            #rxx.append(a), rxy.append(b), rxz.append(c),ryy.append(d), ryz.append(e), rzz.append(f)
            
        if(len(rxx) != nintF+ninlet):
            print"Error- Length not equal"
            
            
        #wall
        wallst=l_wall+4+2
        wallend=wallst+nwall
        
        for line in data0[wallst:wallend]:
            line=line.replace("(","")
            line=line.replace(")","")        
            a, b, c,d,e,f = (item.strip() for item in line.split(' ', 6))
            #rxx.append(a), rxy.append(b), rxz.append(c),ryy.append(d), ryz.append(e), rzz.append(f)
          
                
    
    print 'writing..'
    fp= open("RANS_Pred","w+")
    
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
