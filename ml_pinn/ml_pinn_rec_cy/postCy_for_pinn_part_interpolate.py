#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:49:29 2017

This code process OF data and exports as .pkl to prepData file
for TBNN. prepData reads .pkl and process further

@author: vino
"""
# imports
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy import interpolate
from os import listdir
from os.path import isfile,isdir, join
import cPickle as pickle

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""


fname_1=['cy']

fname_1=np.asarray(fname_1)
fname_1.sort()


# read data from below dir...
path='./'
#path='/home/vino/ml_test/from_nscc_26_dec_2018/foam_run/case_naca_lam'
indir = path



#np.random.seed(1234)
#np.random.shuffle(fname)

fname_2=[]
for i in range(len(fname_1)):
    dir2=indir + '/%s'%fname_1[i]
    tmp=[f for f in listdir(dir2) if isdir(join(dir2, f))]
    fname_2.append(tmp)
    


tmp=[]
foil=[]
for i in range(len(fname_1)):
    for j in range(len(fname_2[i])):
        tmp.append(fname_2[i][j])
        foil.append(fname_2[i][j].split('_')[0])
tmp=np.asarray(tmp)    
foil=np.asarray(foil)

       
coord=[]
#for nn in range(len(foil)):
#    pts=np.loadtxt('./coord_seligFmt_formatted/%s.dat'%foil[nn],skiprows=1)
#    coord.append(pts)
# 

###---around,wake,top,front--------------
L=50
W=50

new_coord=[]
#top
x1=np.linspace(-3,3,L)
y1=np.linspace(3,3,L)

#front
x2=np.linspace(-3,-3,W)
y2=np.linspace(-2.95,2.95,W)

#bot
x3=np.linspace(-3,3,L)
y3=np.linspace(-3,-3,L)

##back-wake - outlet
#x4=np.linspace(3,3,W)
#y4=np.linspace(-2.95,2.95,W)

#back-wake - sampling
x4=np.linspace(1,1,10)
y4=np.linspace(-1,1,10)

tx=0.0
ty=0.05
fx=-0.05
fy=0.0
bx=-0.0
by=-0.05
wx=0.1
wy=0.0

for i in range(5):
    #new_coord.extend(np.asarray([x1+tx*i,y1+ty*i]).transpose())
    #new_coord.extend(np.asarray([x2+fx*i,y2+fy*i]).transpose())
    #new_coord.extend(np.asarray([x3+bx*i,y3+by*i]).transpose())
    new_coord.extend(np.asarray([x4+wx*i,y4+wy*i]).transpose())
    
new_coord=np.asarray(new_coord)
    
'''
co=np.zeros((100,2))

theta=np.linspace(0,360,100)
for i in range(len(theta)):
    co[i,0]= 0.5*np.cos(np.radians(theta[i]))
    co[i,1]= 0.5*np.sin(np.radians(theta[i])) 
    
new_coord=co
'''


plt.figure()
#plt.plot(coord[:,0],coord[:,1],'o')
#plt.plot(x1,y1,x2,y2,x3,y3,x4,y4)
plt.plot(new_coord[:,0],new_coord[:,1],'o')
#plt.xlim([-2,2])
#plt.ylim([-2,2])
plt.show()



#plot
def interp(x,y,Var,new_coord):

    #LinearNDinterpolator
    pD=np.asarray([x,y]).transpose()

    print('interpolation-1...')      
    f1p=interpolate.LinearNDInterpolator(pD,Var)
        
    pu1=np.zeros(len(new_coord))
    for j in range(len(new_coord)):
        pu1[j]=f1p(new_coord[j,0], new_coord[j,1])

    return pu1

aoa=[]
reno=[]
for i in range(len(tmp)):
    reno.append(tmp[i].split('_')[1])    
    aoa.append(tmp[i].split('_')[2])

reno=np.array(map(float, reno))
aoa = np.array(map(float, aoa))


st= [0]
end=[1]


'''np.random.seed(1234534)
mylist=np.random.randint(0,4500,100)

nco=[]
for k in mylist:
    nco.append(coord[k])'''

for jj in range(1):

    myinp_x=[]
    myinp_y=[]
    myinp_para=[]
    myinp_aoa=[]
    myinp_re=[]

    myout_p=[]
    myout_u=[]
    myout_v=[]

    myname=[]
    myfname=[]

    #for ii in mylist:
    for ii in range(st[jj],end[jj]):
        print (ii)
        
        casedir= path +'/%s/%s'%(foil[ii],tmp[ii])
                
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname[:-3].astype(np.int) 
        ymax=int(yname.max())


        x=[]
        with open(casedir +'/%s/ccx'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                x.append(line)
        x = np.array(map(float, x))
       
        y=[]
        with open(casedir +'/%s/ccy'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                y.append(line)
        y = np.array(map(float, y))
        
        z=[]
        with open(casedir +'/%s/ccz'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                z.append(line)
        z = np.array(map(float, z))
        
        p=[]
        with open(casedir +'/%s/p'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                p.append(line)
        p = np.array(map(float, p))
        
        
        # load velocity
        u=[]
        v=[]
        w=[]
        with open(casedir +'/%s/U'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                line=line.replace("(","")
                line=line.replace(")","")        
                a, b, c = (item.strip() for item in line.split(' ', 3))
                u.append(a), v.append(b), w.append(c)
        u = np.array(map(float, u))
        v = np.array(map(float, v))
        w = np.array(map(float, w))
               
        #around
#        I=[]
#        for i in range(len(x)):
#            if (x[i]<=4.5 and x[i]>=-4.5 and y[i]<=4.5 and y[i]>=-4.5 ):
#                I.append(i)
        
#        #front        
#        I=[]
#        for i in range(len(x)):
#            if (x[i]<=-0.5 and x[i]>=-1.0 and y[i]<=0.5 and y[i]>=-0.5 ):
#                I.append(i)        
                
        #top       
#        I=[]
#        for i in range(len(x)):
#            if (x[i]<=0.5 and x[i]>=-0.5 and y[i]<=1.0 and y[i]>=0.5 ):
#                I.append(i)                   
                
        pi=interp(x, y, p, new_coord)
        ui=interp(x, y, u, new_coord)
        vi=interp(x, y, v, new_coord)      
        
      
        
        #plot
        def plot(xp,yp,zp,nc,name):
            plt.figure(figsize=(3, 4))
            #cp = pyplot.tricontour(ys, zs, pp,nc)
            cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
            #cp=pyplot.tricontourf(x1,y1,z1)
            #cp=pyplot.tricontourf(x2,y2,z2)   
            
            #cp = pyplot.tripcolor(xp, yp, zp)
            #cp = pyplot.scatter(ys, zs, pp)
            #pyplot.clabel(cp, inline=False,fontsize=8)
            #plt.xlim(-1,2)
            #plt.ylim(-1,1)    
            plt.axis('off')
            #plt.grid(True)
            #patch.set_facecolor('black')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            #plt.savefig('./plot/%s.eps', format='eps')
            plt.show()
            plt.close()
            
        #plot(x,y,u,20,'name')    
        

        
    #save file
    filepath='./data_file'
      
fp=open('./data_file/cy_sample_x5_10.dat','w')
fp.write('x y p u v :outlet1 - 1 side inlet\n' )
for i in range(len(pi)):
    fp.write('%f %f %f %f %f \n'%(new_coord[i,0], new_coord[i,1], pi[i], ui[i], vi[i]))
fp.close()
    



