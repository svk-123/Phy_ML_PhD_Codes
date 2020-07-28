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
from sklearn.cluster import KMeans

"""
load x y z
data st line: 23 i.e array[22]
only internal points
boundary not loaded: may be required?
"""

# read data from below dir...
path='./foil'
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
    reno.append(tmp[i].split('_')[2])    

reno=np.array(reno)
reno = reno.astype(np.float)


st= [0]
end=[1]


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
    
    my_xc=[]
    my_yc=[]
    my_tt_c=[]
    
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
#        xx=xx[::10,:]
#        xx=xx[-50:]
#        #xx=xx[xx[:,3].argsort()]
#        
        plt.figure(figsize=(10, 4))
        plt.plot(xx[:,0],xx[:,3],'-ob')
        #plt.plot([t1,t10],[xx[:,3].mean(),xx[:,3].mean()],'or')
        plt.xlim([24,28])
        plt.ylim([0.62,1.2])
        #plt.savefig('./plots/%s.png'%ii,format='png',dpi=100)
        plt.show()
        plt.close()
            
        t1=24.0
        t2=25.0
   
#        if (abs(xx[0,0]-xx[1,0]) > 6):
#            t2=xx[2,0]
#            
#        if (t1 > t2):
#            tmp1= t1
#            t1 =t2
#            t2 = tmp1
   
        tt = np.linspace(t1,t2,int (round((t2-t1)/0.05)+1) )
         
    
        mytt = tt-t1
        mytt = mytt
#                   
#        plt.figure(figsize=(3, 4))
#        plt.plot(xx[:,0],xx[:,3],'-ob')
#        #plt.plot([t1,t2],[xx[:,3].mean(),xx[:,3].mean()],'or')
#        plt.savefig('./plots/%s.png'%fname_2[ii],format='png',dpi=100)
#        plt.close()
               
  
        for kk in range(len(tt)):  
            
            ymax=round(tt[kk],2)
            print (ymax)
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
            

            
            #2222                    
            I=[]
            for i in range(len(x)):
                if (x[i]<=1.98 and x[i]>=-0.98 and y[i]<=0.98 and y[i]>=-0.98):
                    I.append(i)   
                                        
            x=x[I]
            y=y[I]
            z=z[I]
            u=u[I]
            v=v[I]
            w=w[I]
            p=p[I]
            
            if(kk==0):
                print(x.shape)
                xy=np.concatenate((x[:,None],y[:,None]),axis=1)
                print (xy.shape)
                kmeans = KMeans(n_clusters=1000, random_state=0).fit(xy)
                c1=kmeans.cluster_centers_
                
                xc=c1[:,0]
                yc=c1[:,1]
            
            
            
            
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
                
            plot(x,y,u,20,'name')    
            
            myinp_x.extend(x)
            myinp_y.extend(y)
            myout_p.extend(p)
            myout_u.extend(u)
            myout_v.extend(v)
        
            tlist=[]
            for k in range(len(x)):
                tlist.append(mytt[kk])
            tlist=np.asarray(tlist)
            
            #original time
            otlist=[]
            for k in range(len(x)):
                otlist.append(tt[kk])
            otlist=np.asarray(otlist)            
                        
            relist=[]
            for k in range(len(x)):
                relist.append(reno[ii])
            relist=np.asarray(relist)
            
            myinp_re.extend(relist)
            myinp_t.extend(tlist)
            otime.extend(otlist)     
            para.append([t1,t2,mytt.max()])

            #for centres
            my_xc.extend(xc)
            my_yc.extend(yc)
            
            tlist_c=[]
            for k in range(len(xc)):
                tlist_c.append(mytt[kk])
            tlist_c=np.asarray(tlist_c)
            my_tt_c.extend(tlist_c)



fp=open('./data_file/naca0012_internal_1211.dat','w')
fp.write('x y t p u v: 24-25: 0.05 dt\n')
for i in range(len(myinp_x)):
    fp.write('%f %f %f %f %f %f\n'%(myinp_x[i], myinp_y[i], myinp_t[i], myout_p[i], myout_u[i], myout_v[i]))
fp.close()

fp=open('./data_file/naca0012_internal_centers_1000_1211.dat','w')
fp.write('x y t : centers for gov. eqn\n')
for i in range(len(my_xc)):
    fp.write('%f %f %f \n'%(my_xc[i], my_yc[i], my_tt_c[i]))
fp.close()
        

        
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
