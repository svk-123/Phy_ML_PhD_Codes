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
import pickle

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""

# read data from below dir...
path='./case'
#path='./cut_from_case_naca_turb'
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
fname_2=fname_2[0]
fname_2=np.asarray(fname_2)

reno=[]
for i in range(len(tmp)):
    reno.append(tmp[i].split('_')[1])    

reno=np.array(reno)
reno = reno.astype(np.float)


st= [0]
end=[1]


'''
###---inout--------------
L=50
W=50

new_coord=[]
x1=np.linspace(-2,3,L)
y1=np.linspace(2,2,L)

x2=np.linspace(-2,-2,W)
y2=np.linspace(-1.95,1.95,W)

x3=np.linspace(-2,3,L)
y3=np.linspace(-2,-2,L)

x4=np.linspace(3,3,W)
y4=np.linspace(-1.95,1.95,W)

tx=0.0
ty=0.05
fx=-0.05
fy=0.0
bx=-0.0
by=-0.05
wx=0.1
wy=0.0

for i in range(1):
    new_coord.extend(np.asarray([x1+tx*i,y1+ty*i]).transpose())
    new_coord.extend(np.asarray([x2+fx*i,y2+fy*i]).transpose())
    new_coord.extend(np.asarray([x3+bx*i,y3+by*i]).transpose())
#    new_coord.extend(np.asarray([x4+wx*i,y4+wy*i]).transpose())
    
new_coord=np.asarray(new_coord)
'''
####-----------------------------------------------------
###---------------wake/around-----------------------------

L=10
W=10

new_coord=[]
x1=np.linspace(-1,1,L)
y1=np.linspace(1,1,L)

x2=np.linspace(-1,-1,W)
y2=np.linspace(-0.95,0.95,W)

x3=np.linspace(-1,1,L)
y3=np.linspace(-1,-1,L)

x4=np.linspace(1,1,W)
y4=np.linspace(-0.95,0.95,W)

tx=0.0
ty=0.05
fx=-0.05
fy=0.0
bx=-0.0
by=-0.05
wx=0.05
wy=0.0

for i in range(4):
    new_coord.extend(np.asarray([x1+tx*i,y1+ty*i]).transpose())
    new_coord.extend(np.asarray([x2+fx*i,y2+fy*i]).transpose())
    new_coord.extend(np.asarray([x3+bx*i,y3+by*i]).transpose())
    new_coord.extend(np.asarray([x4+wx*i,y4+wy*i]).transpose())

new_coord=np.asarray(new_coord)

###############################

###---- On cy wall--------####
co=np.zeros((40,2))
theta=np.linspace(0,360,40)*np.pi/180.

for i in range(len(theta)):
    co[i,0]=0.5*np.cos(theta[i])
    co[i,1]=0.5*np.sin(theta[i])
    
#new_coord=co
######---------------------------------------------------------

plt.figure()
#plt.plot(coord[:,0],coord[:,1],'o')
#plt.plot(x1,y1,x2,y2,x3,y3,x4,y4)
plt.plot(new_coord[:,0],new_coord[:,1],'o')
#plt.xlim([-2,2])
#plt.ylim([-2,2])
plt.show()

####-----------------------------------------------
#interpolate
def interp(x,y,Var,new_coord):

    #LinearNDinterpolator
    pD=np.asarray([x,y]).transpose()

    print('interpolation-1...')      
    f1p=interpolate.LinearNDInterpolator(pD,Var)
        
    pu1=np.zeros(len(new_coord))
    for j in range(len(new_coord)):
        pu1[j]=f1p(new_coord[j,0], new_coord[j,1])

    return pu1
###-------------------------------------------------




for jj in range(1):

    myinp_x=[]
    myinp_y=[]
    myinp_re=[]

    myout_p=[]
    myout_u=[]
    myout_v=[]

    myinp_t=[]
    
    otime=[]
    para=[]
    
    for ii in range(1):
        
        print ( ii)
        
        casedir= path +'/%s/%s'%(fname_1[jj],fname_2[ii])
        print(casedir)
        #need to find max time later....
        yname = [f for f in listdir(casedir) if isdir(join(casedir, f))]
        yname = np.asarray(yname)
        yname.sort()
        yname=yname[:-3].astype(np.float) 
                
        xx=np.loadtxt(casedir+'/postProcessing/forceCoeffs/0/forceCoeffs.dat', skiprows=10)
        xx=xx[::2,:]
        xx=xx[-60:]
        #xx=xx[xx[:,3].argsort()]
        
#        plt.figure(figsize=(10, 4))
#        plt.plot(xx[:,0],xx[:,3],'ob')
#        plt.plot([t1,t10],[xx[:,3].mean(),xx[:,3].mean()],'or')
#        plt.savefig('./plots/%s.png'%ii,format='png',dpi=100)
#        plt.close()
            
        t1=193
        t2=195
        
#        if (abs(xx[0,0]-xx[1,0]) > 6):
#            t2=xx[2,0]
#            
#        if (t1 > t2):
#            tmp1= t1
#            t1 =t2
#            t2 = tmp1
   
        tt = np.linspace(t1,t2,int (round((t2-t1)/0.1)+1) )
         
    
        mytt = tt-t1
        mytt = mytt
                   
        plt.figure(figsize=(3, 4))
        plt.plot(xx[:,0],xx[:,3],'-b')
        #plt.plot([t1,t2],[xx[:,3].mean(),xx[:,3].mean()],'or')
        plt.savefig('./plots/%s.png'%fname_2[ii],format='png',dpi=100)
        plt.close()
               
        
        for kk in range(len(tt)):  
            
            ymax=round(tt[kk],2)
            if((ymax%1) == 0):
                ymax=int(ymax)
            print ('t = ', ymax)
            
            x=[]
            with open(casedir +'/%s/ccx'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    x.append(line)
            x=np.array(x)        
            x = x.astype(np.float)
           
            y=[]
            with open(casedir +'/%s/ccy'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    y.append(line)
            y=np.array(y)        
            y = y.astype(np.float)
            
            z=[]
            with open(casedir +'/%s/ccz'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    z.append(line)
            z=np.array(z)        
            z = z.astype(np.float)
            
            p=[]
            with open(casedir +'/%s/p'%ymax, 'r') as infile:
                data0=infile.readlines()
                npt=int(data0[20])
                for line in data0[22:22+npt]:
                    p.append(line)
            p=np.array(p)        
            p = p.astype(np.float)
            
            
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
                    
            u=np.array(u)        
            u = u.astype(np.float)
            v=np.array(v)        
            v = v.astype(np.float)               
            w=np.array(w)        
            w = w.astype(np.float)       
            
#            #filter wake
#            I=[]
#            for i in range(len(x)):
#                if (x[i]<=1.0 and x[i]>=0.5 and y[i]<=1.0 and y[i]>=-1.0 ):
#                    I.append(i)
            
#            #2222                    
#            I=[]
#            for i in range(len(x)):
#                if (x[i]<=2.98 and x[i]>=-1.98 and y[i]<=1.98 and y[i]>=-1.98):
#                    I.append(i)   
#                                        
#            x=x[I]
#            y=y[I]
#            z=z[I]
#            u=u[I]
#            v=v[I]
#            w=w[I]
#            p=p[I]
            
            pi=interp(x, y, p, new_coord)
            ui=interp(x, y, u, new_coord)
            vi=interp(x, y, v, new_coord) 
            
            
            
            
            if (p.max() > 5):
                print (tmp[ii])
                #fp.write('%s \n'%tmp[ii])
                        
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
                plt.show()
                #plt.savefig('./plotc/%s.eps'%(nname[ii]), format='eps')
                plt.close()
                
            #plot(x,y,u,20,'name')    
            
            myinp_x.extend(new_coord[:,0])
            myinp_y.extend(new_coord[:,1])
            myout_p.extend(pi)
            myout_u.extend(ui)
            myout_v.extend(vi)
    
            tlist=[]
            for k in range(len(new_coord)):
                tlist.append(mytt[kk])
            tlist=np.asarray(tlist)
            
            #original time
            otlist=[]
            for k in range(len(new_coord)):
                otlist.append(tt[kk])
            otlist=np.asarray(otlist)            
                        
            relist=[]
            for k in range(len(new_coord)):
                relist.append(reno[ii])
            relist=np.asarray(relist)
            
            myinp_re.extend(relist)
            myinp_t.extend(tlist)
            otime.extend(otlist)     
            para.append([t1,t2,mytt.max()])



fp=open('./data_file/cy_sample_ar_10x4.dat','w')
fp.write('x y t p u v\n')
for i in range(len(myinp_x)):
    fp.write('%f %f %f %f %f %f\n'%(myinp_x[i], myinp_y[i], myinp_t[i], myout_p[i], myout_u[i], myout_v[i]))
fp.close()

##only-for wall bc
#fp=open('./data_file/cy_xxx.dat','w')
#fp.write('x y t p u v\n')
#for i in range(len(myinp_x)):
#    fp.write('%f %f %f %f 1e-12 1e-12\n'%(myinp_x[i], myinp_y[i], myinp_t[i], myout_p[i]))
#fp.close()

        
#    #save file
#    filepath='./data_file'
#      
#    # ref:[x,y,z,ux,uy,uz,k,ep,nut]
#    info=[' myinp_x, myinp_y, myinp_re, myinp_t, myout_p, myout_u, myout_v, otime, para[t1,t2,mytt.max(), info ']
#
#    data1 = [myinp_x, myinp_y, myinp_re, myinp_t, myout_p, myout_u, myout_v, otime, para, info ]
#
#    with open(filepath+'/cy_un_lam_around_5555_%s.pkl'%(jj+1), 'wb') as outfile1:
#        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)

