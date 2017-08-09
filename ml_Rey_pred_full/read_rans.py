# imports
import os
import glob

import numpy as np
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate
import cPickle as pickle

# rans coordinate
#ccx is modified -2401 () added
def read_rans():
    
    #boundary details for co-ord
    bc=['internalField','inlet','outlet','wall']
    nbc=[477799,2401,2401,39004]
    l_bc=np.zeros(len(bc))
    ist=np.zeros(len(bc))
    iend=np.zeros(len(bc))
    
    #read RANS coord
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
                    for ll in range(nbc[i]):
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
                    for ll in range(nbc[i]):
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
                    for ll in range(nbc[i]):
                        z.append(float(line))            
                else:
                    for line in data0[int(ist[i]):int(iend[i])]:
                        z.append(float(line)) 
        if (len(z)==sum(nbc)):
            print "Z coordinate length correct"
    z = np.array(map(float, z))
    
    #keep only internal field
    print 'x y z - only internal Field stored'
    x=x[0:nbc[0]]
    y=y[0:nbc[0]]
    z=z[0:nbc[0]]
    
    # read RANS data (only internal field)
    print 'reading RANS data - only internal field'
    print 'reading k'
    k=[]
    with open('./rans_data/Re3500/k', 'r') as infile:
        data0=infile.readlines()
        
        for i in range(1):
            line_no=0
            for line in data0:
                if bc[i] in line:
                    l_bc[i]=line_no
    
                    for tmp in range(line_no,line_no+5):
                        if str(nbc[i]) in data0[tmp]:
                            ist[i]=tmp+2
                            iend[i]=tmp+2+nbc[i]
                
                line_no=line_no+1
        for i in range(1):
                if(ist[i]==iend[i]):
                    for ll in range(nbc[i]):
                        k.append(float(line))            
                else:
                    for line in data0[int(ist[i]):int(iend[i])]:
                        k.append(float(line)) 
        if (len(k)==nbc[0]):
            print "k - length correct -only internal field"
    k = np.array(map(float, k))
    

    print 'reading ep'
    ep=[]
    with open('./rans_data/Re3500/epsilon', 'r') as infile:
        data0=infile.readlines()
        
        for i in range(1):
            line_no=0
            for line in data0:
                if bc[i] in line:
                    l_bc[i]=line_no
    
                    for tmp in range(line_no,line_no+5):
                        if str(nbc[i]) in data0[tmp]:
                            ist[i]=tmp+2
                            iend[i]=tmp+2+nbc[i]
                
                line_no=line_no+1
        for i in range(1):
                if(ist[i]==iend[i]):
                    for ll in range(nbc[i]):
                        ep.append(float(line))            
                else:
                    for line in data0[int(ist[i]):int(iend[i])]:
                        ep.append(float(line)) 
        if (len(ep)==nbc[0]):
            print "ep - length correct -only internal field"
    ep = np.array(map(float, ep))
    
    print 'reading nut'
    nut=[]
    with open('./rans_data/Re3500/nut', 'r') as infile:
        data0=infile.readlines()
        
        for i in range(1):
            line_no=0
            for line in data0:
                if bc[i] in line:
                    l_bc[i]=line_no
    
                    for tmp in range(line_no,line_no+5):
                        if str(nbc[i]) in data0[tmp]:
                            ist[i]=tmp+2
                            iend[i]=tmp+2+nbc[i]
                
                line_no=line_no+1
        for i in range(1):
                if(ist[i]==iend[i]):
                    for ll in range(nbc[i]):
                        nut.append(float(line))            
                else:
                    for line in data0[int(ist[i]):int(iend[i])]:
                        nut.append(float(line)) 
        if (len(nut)==nbc[0]):
            print "nut - length correct -only internal field"
    nut = np.array(map(float, nut))
    
    
    # velocity grad
    print 'reading vel.grad'
    ux=[]
    uy=[]
    uz=[]
    vx=[]
    vy=[]
    vz=[]
    wx=[]
    wy=[]
    wz=[]
    with open('./rans_data/Re3500/grad(U)', 'r') as infile:
        data0=infile.readlines()
        
        for i in range(1):
            line_no=0
            for line in data0:
                if bc[i] in line:
                    l_bc[i]=line_no
    
                    for tmp in range(line_no,line_no+5):
                        if str(nbc[i]) in data0[tmp]:
                            ist[i]=tmp+2
                            iend[i]=tmp+2+nbc[i]
                
                line_no=line_no+1
        for i in range(1):
                if(ist[i]==iend[i]):
                    for ll in range(nbc[i]):
                        line=data0[ll]
                        line=line.replace("(","")
                        line=line.replace(")","")        
                        a, b, c,d,e,f,g,h,i = (item.strip() for item in line.split(' ', 9))
                        ux.append(a), uy.append(b), uz.append(c),vx.append(d), vy.append(e), vz.append(f),\
                        wx.append(g), wy.append(h), wz.append(i)           
                else:
                    for line in data0[int(ist[i]):int(iend[i])]:
                        line=line.replace("(","")
                        line=line.replace(")","")        
                        a, b, c,d,e,f,g,h,i = (item.strip() for item in line.split(' ', 9))
                        ux.append(a), uy.append(b), uz.append(c),vx.append(d), vy.append(e), vz.append(f),\
                        wx.append(g), wy.append(h), wz.append(i) 
                            
    ux = np.array(map(float, ux))
    uy = np.array(map(float, uy))
    uz = np.array(map(float, uz))
    vx = np.array(map(float, vx))
    vy = np.array(map(float, vy))
    vz = np.array(map(float, vz))
    wx = np.array(map(float, wx))
    wy = np.array(map(float, wy))
    wz = np.array(map(float, wz))
    if (len(ux)==len(uy)==len(uz)==len(vx)==len(vy)==len(vz)==len(wx)==len(wy)==len(wz)==nbc[0]):
        print "vel.grad- length correct"        
        
        
    #rans calculations
    print 'rans data processing...'
         
    # calculation No Change Requirred
    #uiuj calculation
    uu=-nut*(ux+ux) + (2./3.)*k
    uv=-nut*(uy+vx)
    uw=-nut*(uz+wx)
    vv=-nut*(vy+vy) + (2./3.)*k
    vw=-nut*(vz+wy)
    ww=-nut*(wz+wz) + (2./3.)*k
    
    # load to single variable
    UUR=np.zeros((len(z),6))
    UUR[:,0]=uu
    UUR[:,1]=uv
    UUR[:,2]=uw
    UUR[:,3]=vv
    UUR[:,4]=vw
    UUR[:,5]=ww
    
    # anisotropy tensor
    aR=np.zeros((len(z),6))
    aR[:,0]=uu-(2./3.)*k
    aR[:,1]=uv
    aR[:,2]=uw
    aR[:,3]=vv-(2./3.)*k
    aR[:,4]=vw
    aR[:,5]=ww-(2./3.)*k
    
     #non dim. anisotropy tensor
    bR=np.zeros((len(z),6))  
    for i in range(len(z)):
        bR[i,:]=aR[i,:]/(2*k[i])        
            
            
    dU=np.zeros((len(ux),3,3))
    dUT=np.zeros((len(ux),3,3))
    for i in range(len(ux)):
        dU[i,0,0]=ux[i]
        dU[i,0,1]=uy[i]
        dU[i,0,2]=uz[i]
        dU[i,1,0]=vx[i]
        dU[i,1,1]=vy[i]
        dU[i,1,2]=vz[i]
        dU[i,2,0]=wx[i]
        dU[i,2,1]=wy[i]
        dU[i,2,2]=wz[i]
        
    for i in range(len(ux)):
        dUT[i,0,0]=ux[i]
        dUT[i,0,1]=vx[i]
        dUT[i,0,2]=wx[i]
        dUT[i,1,0]=uy[i]
        dUT[i,1,1]=vy[i]
        dUT[i,1,2]=wy[i]
        dUT[i,2,0]=uz[i]
        dUT[i,2,1]=vz[i]
        dUT[i,2,2]=wz[i]    
        
    S=0.5*(dU+dUT)
    R=0.5*(dU-dUT)    
    
    # Non dim S,R
    for i in range(len(ux)):
        S[i,:,:]=S[i,:,:]*(k[i]/ep[i])
        R[i,:,:]=R[i,:,:]*(k[i]/ep[i])    
        
           
    TrS2  =np.zeros(len(ux))
    TrR2  =np.zeros(len(ux))
    TrS3  =np.zeros(len(ux))
    TrR2S =np.zeros(len(ux))
    TrR2S2=np.zeros(len(ux))
    TrSR2 =np.zeros(len(ux))
    TrS2R2=np.zeros(len(ux))
    
    for i in range(len(ux)):   
        TrS2[i]=np.trace(np.matmul(S[i,:,:],S[i,:,:]))
        TrR2[i]=np.trace(np.matmul(R[i,:,:],R[i,:,:]))
        TrS3[i]=np.trace( np.matmul(np.matmul(S[i,:,:],S[i,:,:]),S[i,:,:]) )
        TrR2S[i]=np.trace( np.matmul(np.matmul(R[i,:,:],R[i,:,:]),S[i,:,:]) )    
        TrR2S2[i]=np.trace( np.matmul(np.matmul(R[i,:,:],R[i,:,:]),np.matmul(S[i,:,:],S[i,:,:])) )       
        TrSR2[i]=np.trace( np.matmul(S[i,:,:],np.matmul(R[i,:,:],R[i,:,:])) )     
        TrS2R2[i]=np.trace( np.matmul(np.matmul(S[i,:,:],S[i,:,:]),np.matmul(R[i,:,:],R[i,:,:])) )       
        
    I=np.zeros((len(ux),3,3))   
    for i in range(len(ux)):
        I[i,:,:]=np.eye(3)
        
    T1=np.zeros((len(ux),3,3))
    T2=np.zeros((len(ux),3,3))
    T3=np.zeros((len(ux),3,3))
    T4=np.zeros((len(ux),3,3))
    T5=np.zeros((len(ux),3,3))
    T6=np.zeros((len(ux),3,3))
    T7=np.zeros((len(ux),3,3))
    T8=np.zeros((len(ux),3,3))
    T9=np.zeros((len(ux),3,3))
    T10=np.zeros((len(ux),3,3))
    
    for i in range(len(ux)):     
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
    
    L=np.zeros((len(z),5))
    for i in range(len(z)):
        L[i,0]=L1[i]
        L[i,1]=L2[i]
        L[i,2]=L3[i]
        L[i,3]=L4[i]
        L[i,4]=L5[i]
        
    #Components-6 Input for ML
    T1c=np.zeros((len(ux),6))
    T2c=np.zeros((len(ux),6))
    T3c=np.zeros((len(ux),6))
    T4c=np.zeros((len(ux),6))
    T5c=np.zeros((len(ux),6))
    T6c=np.zeros((len(ux),6))
    T7c=np.zeros((len(ux),6))
    T8c=np.zeros((len(ux),6))
    T9c=np.zeros((len(ux),6))
    T10c=np.zeros((len(ux),6))
    
    for i in range(len(ux)):
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
        
        
        
    Tc=np.zeros((len(z),10,6))
    for i in range(len(ux)):
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
    T1m=np.zeros((len(z),10))
    T2m=np.zeros((len(z),10))
    T3m=np.zeros((len(z),10))
    T4m=np.zeros((len(z),10))
    T5m=np.zeros((len(z),10))
    T6m=np.zeros((len(z),10))
    
    for i in range(len(ux)):
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
        
    for i in range(len(ux)):
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
        
    for i in range(len(ux)):
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
        
    for i in range(len(ux)):
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
        
    for i in range(len(ux)):
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
        
    for i in range(len(ux)):
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
        
    return (x,z,y,k,L,T1m,T2m,T3m,T4m,T5m,T6m) 
    print 'done processing rans'   
        
        
        
        
        