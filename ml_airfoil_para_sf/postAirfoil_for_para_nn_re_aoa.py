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

# read data from below dir...
path='../cnn_airfoil_sf/OF_results/case_re_aoa_2'

indir = path

fname_1 = [f for f in listdir(indir) if isdir(join(indir, f))]
fname_1.sort()
fname_1=np.asarray(fname_1)

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

#foilR=['naca23012xx','naca66018xx']
#ind_del=[]
#for i in range(2):
#    if foilR[i] in tmp:
#        ind=np.argwhere(tmp==foilR[i])
#        ind_del.extend(ind)
#fname=np.delete(fname,ind_del,0)
       
coord=[]
for nn in range(len(foil)):
    pts=np.loadtxt('../cnn_airfoil_sf/airfoil_data/coord_seligFmt_formatted/%s.dat'%foil[nn],skiprows=1)
    coord.append(pts)
 
datafile='./data_file/param_216_16.pkl'
with open(datafile, 'rb') as infile:
    result = pickle.load(infile)
para=result[0][0]    
pname=result[1]
pname=np.asarray(pname)

aoa=[]
reno=[]
for i in range(len(tmp)):
    reno.append(tmp[i].split('_')[1])    
    aoa.append(tmp[i].split('_')[2])

reno=np.array(map(float, reno))
aoa = np.array(map(float, aoa))

my_para=[]
for i in range(len(foil)):
    if foil[i] in pname:
        ind=np.argwhere(pname==foil[i])
        my_para.append(para[int(ind)])

    else:
        print('not in pname %s'%foil[i])



st=[0,500,1000]
end=[500,1000,len(foil)]
fp=open('foil_error.dat','w+')

for jj in range(3):


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

    for ii in range(st[jj],end[jj]):
        print ii
        
        casedir= path +'/%s/%s'%(foil[ii],tmp[ii])
                
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname[:-2].astype(np.int) 
        ymax=int(yname.max())
        
        x=[]
        with open(casedir +'/%s/Cx'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                x.append(line)
        x = np.array(map(float, x))
       
        y=[]
        with open(casedir +'/%s/Cy'%ymax, 'r') as infile:
            data0=infile.readlines()
            npt=int(data0[20])
            for line in data0[22:22+npt]:
                y.append(line)
        y = np.array(map(float, y))
        
        z=[]
        with open(casedir +'/%s/Cz'%ymax, 'r') as infile:
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
               
        
        
        #filter within xlim,ylim
        I=[]
        for i in range(len(x)):
            if (x[i]<=2.2 and x[i]>=-0.6 and y[i]<=0.6 and y[i]>=-0.6 ):
                I.append(i)
                
                
        x=x[I]
        y=y[I]
        z=z[I]
        u=u[I]
        v=v[I]
        w=w[I]
        p=p[I]
        
        if (p.max() > 2):
            print tmp[ii]
            fp.write('%s \n'%tmp[ii])
        
        
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
            plt.xlim(-1,2)
            plt.ylim(-1,1)    
            plt.axis('off')
            #plt.grid(True)
            #patch.set_facecolor('black')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            #plt.savefig('./plotc/%s.eps'%(nname[ii]), format='eps')
            plt.close()
            
        #plot(x,y,u,20,'name')    
        
        myinp_x.extend(x)
        myinp_y.extend(y)
        myout_p.extend(p)
        myout_u.extend(u)
        myout_v.extend(v)

        paralist=[]
        for k in range(len(x)):
            paralist.append(my_para[ii])
        paralist=np.asarray(paralist)
    
        aoalist=[]
        for k in range(len(x)):
            aoalist.append(aoa[ii])
        aoalist=np.asarray(aoalist)
        
        relist=[]
        for k in range(len(x)):
            relist.append(reno[ii])
        relist=np.asarray(relist)
        
        namelist=[]
        for k in range(len(x)):
            namelist.append(foil[ii])
        namelist=np.asarray(namelist)

        #fnamelist=[]
        #for k in range(len(x)):
        #    fnamelist.append(tmp[ii])
        #fnamelist=np.asarray(fnamelist)

        myinp_para.extend(paralist)
        myinp_aoa.extend(aoalist)
        myinp_re.extend(relist)
        myname.extend(namelist)
        #myfname.extend(fnamelist)       
        
    #save file
    filepath='./data_file'
      
    # ref:[x,y,z,ux,uy,uz,k,ep,nut]
    info=['myinp_x, myinp_y, myinp_para, myinp_re, myinp_aoa, myout_p, myout_u, myout_v, coord, mynae, info']

    data1 = [myinp_x, myinp_y, myinp_para, myinp_re, myinp_aoa, myout_p, myout_u, myout_v, coord, myname, info ]

    with open(filepath+'/foil_aoa_nn_p16_ph_1_tr_2%sxx.pkl'%(jj+1), 'wb') as outfile1:
        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)

    
fp.close()        
