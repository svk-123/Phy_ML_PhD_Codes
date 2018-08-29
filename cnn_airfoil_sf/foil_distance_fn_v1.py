#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:09:04 2017

@author: vinoth
"""
#from __future__ import print_function

import time
start_time = time.time()

# Python 3.5
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from skimage import io, viewer,util 
from scipy import interpolate
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy.linalg as LA
import cPickle as pickle
import pandas
from skimage import io, viewer,util 
np.set_printoptions(threshold=np.inf)

path='./airfoil_data'

indir=path+'/foil200_aoa'

fname = [f for f in listdir(indir) if isfile(join(indir, f))]
fname.sort()

nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])

img_mat=[]
img_mat_n=[]
coord_mat=[]

#pixel calc
a=72
zl=108
b=108
px=1/72.
    
xu_n=np.linspace(a,2*a-1,a)
xl_n=xu_n
xu_n=xl_n[::-1]
    
xi=np.concatenate((xu_n,xl_n),axis=0)
xi=xi.astype(int)

tot_bor=[]
tot_ins=[]

for i in range(3,4):
    print i
    pts=np.loadtxt(indir+'/%s'%fname[i],skiprows=1)
    
    
    figure=plt.figure(figsize=(3,4))
    plt0, =plt.plot(pts[:,0],pts[:,1],'k',linewidth=1e-6,label='true')
    #plt0, =plt.plot((xu_n-a)/(a-1),uy,linewidth=2,label='true')
    #plt0, =plt.plot((xl_n-a)/(a-1),ly,linewidth=2,label='true')
    
    #plt0, =plt.plot(up_x,lr_x,linewidth=2,label='true')
    #plt0, =plt.plot(up_y,lr_y,linewidth=2,label='true')
        
    plt.xlim(-1,2)
    plt.ylim(-1,1)    
    plt.axis('off')
    
    #plt.grid(True)
    #patch.set_facecolor('black')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.savefig('./plot/%d_%s.eps'%(i,nname[i]), format='eps')
    #plt.show() 
    plt.close()
               
    img_foil = io.imread('./plot/%d_%s.eps'%(i,nname[i]), as_grey=True)  # load the image as grayscale
    img_mat.append(img_foil)
    print('image matrix size: ', img_foil.shape )     # print the size of image
        
    bor=np.argwhere(img_foil==0.0)
    bor=np.asarray(bor)
    tot_bor.append(bor)
    
    # outer-innner both
    img_foil_n=img_foil.copy()
    
    
    
    for m in range(img_foil.shape[0]):
        for n in range(img_foil.shape[1]):
            if (([m,n] == bor).all(1).any() == False):
         
                dist=LA.norm((bor-[m,n]),axis=1).min()
                
                img_foil_n[m,n]=dist*0.01
        
    
    #inner
    xi=bor[:,1].copy()
    yi=bor[:,0].copy()
    tmp=np.concatenate((xi[:,None],yi[:,None]),axis=1)
    tmp=tmp[tmp[:,1].argsort()]
    tmp=tmp[tmp[:,0].argsort(kind='mergesort')]
    
    unique, counts = np.unique(tmp[:,0], return_counts=True)
    
    if (np.any(counts==1) == True):
        ind1=np.argwhere(counts==1)
        raise ValueError('singe Value exits Error')
        
    if (np.any(counts%2 ==1) == True):    
        ind2=np.argwhere(counts%2 ==1)
        for j in range(len(ind2)):
            
            num=unique[ind2[j]]
            ind3 = np.argwhere(tmp[:,0]==num)
            tmpval=tmp[ind3,1]
    
            for k in range(len(tmpval)-1):
                if(tmpval[k]+1==tmpval[k+1]):
                    if(tmpval[k]==tmpval.min()):
                        tmp=np.delete(tmp,ind3[k],0)
                    if(tmpval[k+1]==tmpval.max()):
                        tmp=np.delete(tmp,ind3[k+1],0)
                        
    xtmp=[]
    ytmp=[]
    ins=[]
    for j in range(0,len(tmp),2):
        if(tmp[j,1] != tmp[j+1,1]):
            tmp1=range(min(tmp[j,1],tmp[j+1,1]),max(tmp[j,1],tmp[j+1,1])+1)
            for k in range(len(tmp1)):
                xtmp.append(tmp[j,0])
                ytmp.append(tmp1[k])
                ins.append([tmp[j,0],tmp1[k]])
    xtmp=np.asarray(xtmp)            
    ytmp=np.asarray(ytmp)
    tot_ins.append(ins)
        
    for m in range(len(xtmp)):
        
        if (([ ytmp[m],xtmp[m] ] == bor).all(1).any() == False):
         
            dist=LA.norm((bor-[ ytmp[m],xtmp[m] ]),axis=1).min()
                
            img_foil_n[ ytmp[m], xtmp[m] ]= -1*dist*0.01

     
    
    
    #img_foil_n[ytmp,xtmp]=2.0 # for visibility
    img_foil_n[bor[:,0],bor[:,1]]=0.0 # for visible plotting chnage to 1.0
    if(img_foil_n[yi[0],xi[0]]!=0):
        raise ValueError('Error')
    img_mat_n.append(img_foil_n)
        
    plt.imshow(img_foil_n,aspect=3/4.)
    #plt.xlim(70,145)
    #plt.ylim(155,135) 
    #plt.yticks(np.arange(0,288, 2.0))
    #plt.grid(linewidth=1e-3, linestyle='-',alpha=0.1)
    #plt.axis().set_aspect(3./4)
    plt.savefig('./plot_df/%d_%s.eps'%(i,nname[i]), format='eps',dpi=300)
    plt.close()
    #plt.axis('off')
        
    '''plt.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    xp, yp = np.meshgrid(range(216), range(216))
    cp = plt.contourf(xp,yp[::-1],img_foil_n,10,cmap=cm.jet)
    #v= np.linspace(0, 0.05, 15, endpoint=True)
    #cp = plt.tricontourf(xp,yp,zp,v,cmap=cm.jet,extend='both')
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #plt.clabel(cp, inline=True,fontsize=8)
    plt.colorbar()
    #plt.title('%s  '%flist[ii]+name)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    plt.savefig('./plot_df/%s.png'%(nname[i]), format='png',bbox_inches='tight', dpi=100)
    plt.show()'''
    
data2=[coord_mat,img_mat,img_mat_n,tot_bor,tot_ins,nname,'0-coord,1-imag,2-img_df,3-bor(yx),4-ins(xy)r,5-nname']
with open(path+'/foil_aoa_df.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)
    
    

