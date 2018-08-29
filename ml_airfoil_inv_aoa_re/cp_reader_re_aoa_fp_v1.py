#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

@author: vinoth
"""

import time
start_time = time.time()

# Python 3.5
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from skimage import io, viewer,util 

import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy import interpolate
import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)
import xlrd

path='./foil_all_re_aoa'
renomax=300000
aoamax=12

#path='./'

for p in range(7,9):
    
    fn='/Cp_Graph_%d.xlsx'%p
    sn='Cp_Graph_%d'%p
    
    excel_sheet = xlrd.open_workbook(path+fn)
    sheet1= excel_sheet.sheet_by_name(sn)
           
    row = sheet1.row(0)   
    row=np.asarray(row)
    for i in range(len(row)):
        row[i]=row[i].value.encode('ascii','ignore')
    
    name=[]
    reno=[]
    aoa=[]
    for i in range(len(row)):
        if 'Re' in row[i]:
            name.append(row[i])
            reno.append(row[i])
            aoa.append(row[i])
            
    for i in range(len(name)):
        name[i]=name[i].split('-Re')[0]

    
    for i in range(len(reno)):
        reno[i]=reno[i].split('Re=')[1]
        reno[i]=reno[i].split('-Alpha')[0]
        reno[i]=reno[i].strip()
    reno=np.asarray(reno)
    reno=reno.astype(np.float)
    reno=reno/renomax
    
    for i in range(len(aoa)):
        aoa[i]=aoa[i].split('-NCrit')[0]
        aoa[i]=aoa[i].split('Alpha=')[1]     
        aoa[i]=aoa[i].strip()
    aoa=np.asarray(aoa)
    aoa=aoa.astype(np.float)
    aoa=aoa/aoamax
        
    coord=[]
    for i in range(len(name)):
        coord.append(np.loadtxt(path+'/foil_part_%d/%s.dat'%(p,name[i]),skiprows=1))
    
    
    tmp = pd.read_excel(path+fn,sep=",",delimiter=",",header=None,skiprows=1)
    tmp=np.asarray(tmp) 
       
    n=tmp.shape[1]
    
    #check
    if (len(name) !=(n/3)):
        raise ValueError('length not matching')
    
    i=0
    val=range(n)
    I=[]
    for i in range(len(val)-1):
        if( ((val[i]+1)%3) != 0 ):
            I.append(i)
             
    tmp=tmp[:,I]
    
    tmp2=[]
    for j in range(tmp.shape[1]):
        if(j%2==0):
            tmp11=[]
            tmp12=[]
            for i in range(tmp[:,j].shape[0]):
                if (pd.isnull(tmp[i,j])==False and tmp[i,j]!= u' '):
                    tmp11.append(tmp[i,j])
                    tmp12.append(tmp[i,j+1])
            tmp2.append(np.concatenate((np.asarray(tmp11)[:,None],np.asarray(tmp12)[:,None]),axis=1))
       
    del excel_sheet
    del sheet1
    del row
    del tmp
    del tmp11
    del tmp12
     
    img_mat_up=[]
    img_mat_lr=[]
    
    cp_up=[]
    cp_lr=[]
        
    for i in range(len(tmp2)):
        
        print i
        
        n=tmp2[i].shape[0]
        indx=np.argmin(tmp2[i][:,0])

        up=tmp2[i][:indx+1,:]
        lr=tmp2[i][indx:,:]
        
        cp_up.append(up)
        cp_lr.append(lr)
        
        figure=plt.figure(figsize=(2,4))
        plt0, =plt.plot(up[:,0],up[:,1],'k',linewidth=2,label='true')
        #plt0, =plt.plot(lr[:,0],lr[:,1],'b',linewidth=2,label='true')
        #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
        #plt.legend(fontsize=16)
        #plt.xlabel('alpha',fontsize=16)
        #plt.ylabel('cl',fontsize=16)
        #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
        plt.xlim(-0.05,1.05)
        plt.ylim(-10.0,1.0)    
        plt.axis('off')
        #plt.grid(True)
        #patch.set_facecolor('black')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.savefig('./plot1/up_%s.eps'%i, format='eps')
        #plt.show() 
        plt.close()
        
        figure=plt.figure(figsize=(2,4))
        plt0, =plt.plot(lr[:,0],lr[:,1],'k',linewidth=2,label='true')
        #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
        #plt.legend(fontsize=16)
        #plt.xlabel('alpha',fontsize=16)
        #plt.ylabel('cl',fontsize=16)
        #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
        plt.xlim(-0.05,1.05)
        plt.ylim(-1.0,1.0)    
        plt.axis('off')
        #plt.grid(True)
        #patch.set_facecolor('black')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.savefig('./plot1/lr_%s.eps'%i, format='eps')
        #plt.show() 
        plt.close() 
        
        img_up = io.imread('./plot1/up_%s.eps'%i, as_grey=True)  # load the image as grayscale
        img_up = util.invert(img_up)
        
        img_up[0,:]=reno[i]
        img_up[-1:,:]=reno[i]
        img_up[:,0]=reno[i]
        img_up[:,-1:]=reno[i]   
        
        img_up[1,:]=aoa[i]
        img_up[-2:-1,:]=aoa[i]
        img_up[:,1]=aoa[i]
        img_up[:,-2:-1]=aoa[i]  
        
        img_mat_up.append(img_up)
        print 'image matrix size: ', img_up.shape      # print the size of image
        
        img_lr = io.imread('./plot1/lr_%s.eps'%i, as_grey=True)  # load the image as grayscale
        img_lr = util.invert(img_lr)
        
        img_lr[0,:]=reno[i]
        img_lr[-1:,:]=reno[i]
        img_lr[:,0]=reno[i]
        img_lr[:,-1:]=reno[i]   
        
        img_lr[1,:]=aoa[i]
        img_lr[-2:-1,:]=aoa[i]
        img_lr[:,1]=aoa[i]
        img_lr[:,-2:-1]=aoa[i]  
        
        img_mat_lr.append(img_lr)
        print 'image matrix size: ', img_lr.shape      # print the size of image
        
    xx=np.loadtxt(path+'/xx.txt')   
    img_mat=[]
    
    for i in range(len(coord)):
        
        l=len(coord[i])
        ind=np.argmin(coord[i][:,0])
        
        up_x=coord[i][:ind+1,0]
        up_y=coord[i][:ind+1,1]
        
        lr_x=coord[i][ind:,0]
        lr_y=coord[i][ind:,1]    
        
        up_x[0]=1
        up_x[-1:]=0
        
        lr_x[0]=0    
        lr_x[-1:]=1
        
        fu = interpolate.interp1d(up_x, up_y)
        u_yy = fu(xx)
        
        fl = interpolate.interp1d(lr_x, lr_y)
        l_yy = fl(xx)   
                
        yout=np.zeros(len(u_yy)*2)
        yout[0:len(xx)]=u_yy
        yout[len(xx):]=l_yy    
        img_mat.append(yout)
        
        #plot
        '''figure=plt.figure(figsize=(6,5))
        plt0, =plt.plot(coord[i][:,0],coord[i][:,1],'o',linewidth=2,label='true')
        plt0, =plt.plot(xx,u_yy)
        plt0, =plt.plot(xx,l_yy)    
        #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
        #plt.legend(fontsize=16)
        #plt.xlabel('alpha',fontsize=16)
        #plt.ylabel('cl',fontsize=16)
        #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
        plt.xlim(0,1.)
        plt.ylim(-0.18,0.18)    
        plt.axis('off')
        plt.show()''' 
        
    info='img_mat_up, img_mat_lr, img_mat, reno, aoa, xx, name, cp_up, cp_lr, info]'
    data1=[img_mat_up,img_mat_lr,img_mat,reno,aoa,xx,name,cp_up,cp_lr,info]
    with open(path+'/data_re_aoa_fp_%d.pkl'%p, 'wb') as outfile:
        pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)
        

    

