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
from scipy import interpolate
import pandas as pd
from os import listdir
from os.path import isfile, join

import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)


path='./airfoil_1600_1aoa_1re/'

import xlrd
excel_sheet = xlrd.open_workbook(path+'/Cp_Graph.xlsx')
sheet1= excel_sheet.sheet_by_name('Cp_Graph')
       
row = sheet1.row(0)   
row=np.asarray(row)

for i in range(len(row)):
    row[i]=row[i].value.encode('ascii','ignore')
name=[]

for i in range(len(row)):
    if 'Re' in row[i]:
        name.append(row[i])

for i in range(len(name)):
    name[i]=name[i].split('-Re')[0]
    
   
coord=[]
for i in range(len(name)):
    coord.append(np.loadtxt(path+'coord_seligFmt_formatted/%s.dat'%name[i],skiprows=1))


tmp = pd.read_excel(path+'/Cp_Graph.xlsx',sep=",",delimiter=",",header=None,skiprows=1)
tmp=np.asarray(tmp) 
   
n=tmp.shape[1]


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



img_mat_up=[]
img_mat_lr=[]

up_mm=[]
lr_mm=[]

#not good plot
unname=np.genfromtxt('airfoil_plot_not_good',dtype='str')

xx=np.loadtxt('xx.txt')
cp_mat=[]
for i in range(len(tmp2)):
#for i in range(10):
    print i
    if name[i] not in unname:
        n=tmp2[i].shape[0]
        if(n%2 == 0):
            up=tmp2[i][:(n/2),:]
            lr=tmp2[i][(n/2):,:]
        else:
            up=tmp2[i][:(n/2)+1,:]
            lr=tmp2[i][(n/2):,:]
    
        up_x=up[:,0]
        up_y=up[:,1]
            
        lr_x=lr[:,0]
        lr_y=lr[:,1]

        up_x[0]=1
        up_x[-1:]=0            
        
        lr_x[0]=0
        lr_x[-1:]=1   
        
        fu = interpolate.interp1d(up_x, up_y)
        cp_up = fu(xx)
        
        fl = interpolate.interp1d(lr_x, lr_y)
        cp_lr = fl(xx)   
        
        
        yout=np.zeros(len(cp_up)*2)
        yout[0:len(xx)]=cp_up
        yout[len(xx):]=cp_lr 
        cp_mat.append(yout)
        
        #plot
        figure=plt.figure(figsize=(6,4))
        plt0, =plt.plot(up_x,up_y,'og',linewidth=2,label='true')
        plt0, =plt.plot(lr_x,lr_y,'or',linewidth=2,label='true')
        plt0, =plt.plot(xx,cp_up)
        plt0, =plt.plot(xx,cp_lr)    
        #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
        #plt.legend(fontsize=16)
        #plt.xlabel('alpha',fontsize=16)
        #plt.ylabel('cl',fontsize=16)
        #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
        #plt.xlim(-0.05,1.05)
        #plt.ylim(-0.18,0.18)    
        #plt.axis('off')
        plt.savefig('./plot/%s.png'%name[i])
        plt.show() 
        print i

     
    
 
img_mat=[]
for i in range(len(coord)):
#for i in range(10):
    print i
    if name[i] not in unname:  
        #plot
        figure=plt.figure(figsize=(2,2))
        plt0, =plt.plot(coord[i][:,0],coord[i][:,1],'k',linewidth=2,label='true')
        #plt1, =plt.plot(val_inp[:,4],out,'-or',linewidth=2,label='nn')  
        #plt.legend(fontsize=16)
        #plt.xlabel('alpha',fontsize=16)
        #plt.ylabel('cl',fontsize=16)
        #plt.title('NACA%sRe=%se6'%(name[i],rey_no[i]),fontsize=16)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=False, shadow=False)
        plt.xlim(0,1.)
        plt.ylim(-0.18,0.18)    
        plt.axis('off')
        #plt.grid(True)
        #patch.set_facecolor('black')
        plt.savefig('./plot/coord_%s'%name[i], format='png')
        plt.show() 
    
        img = io.imread('./plot/coord_%s'%name[i], as_grey=True)  # load the image as grayscale
        img = util.invert(img)
        img_mat.append(img)
        print 'image matrix size: ', img.shape      # print the size of image
        #print '\n First 5 columns and rows of the image matrix: \n', img[150:210,170:180] 
        #viewer.ImageViewer(img).show()  
        #img=img-1
        #img=abs(img)
        #viewer.ImageViewer(img).show()
    


nname=[]
for kk in range(len(name)):
    if name[kk] not in unname: 
        nname.append(name[kk])



'''data1=[img_mat_up,img_mat_lr,img_mat,xx,nname]
with open(path+'data_cp_fp_1600.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)'''
    


data2=[cp_mat,img_mat,xx,nname]
with open(path+'foil_cp_s144_1343.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)
    
    
    
    
    

