__author__ = 'ryanbarr'

import numpy as np
import matplotlib.pylab as plt
from kulfan_to_coord import CST_shape
from pyOpt import Optimization, SLSQP
import os, sys, time
import pdb
from os import listdir
from os.path import isfile, isdir, join


#load file name
casedir='./picked_uiuc_101/'
fname = [f for f in listdir(casedir) if isfile(join(casedir, f))]
fname.sort()

nname=[]
for i in range(len(fname)):
    nname.append(fname[i].split('.dat')[0])

fp1=open('foilname.dat','w')
fp2=open('cst.dat','w')


for ii in range(1433):
    
    print(ii)
    
    foil=np.loadtxt('./picked_uiuc_101/%s.dat'%nname[ii])
    
    x1 = foil[:,0]
    y1 = foil[:,1]
    
    global airfoil_CST
    dz = 0
    N = 101
    
    def objfunc(x):
    
        wl = [x[0], x[1], x[2], x[3]]
        wu = [x[4], x[5], x[6], x[7]]
    
        global airfoil_CST
        airfoil_CST = CST_shape(wl, wu, dz, N)
        coordinates = airfoil_CST.inv_airfoil_coor(x1)
        x2 = coordinates[:][0]
        y2 = coordinates[:][1]
    
        f = 0
        for i in range(N-1):
            f += abs(y1[i]*100 - y2[i]*100)**2
    
        g = []
    
        fail = 0
        return f, g, fail
    
    
    # =============================================================================
    #
    # =============================================================================
    opt_prob = Optimization('CST Parameterization', objfunc)
    opt_prob.addVar('x1','c', lower=-2.0,upper=2.0, value=-1.0)
    opt_prob.addVar('x2','c', lower=-2.0,upper=2.0, value=-1.0)
    opt_prob.addVar('x3','c', lower=-2.0,upper=2.0, value=-1.0)
    opt_prob.addVar('x4','c', lower=-2.0,upper=2.0, value=-1.0)
    opt_prob.addVar('x5','c', lower=-2.0, upper=2.0, value=1.0)
    opt_prob.addVar('x6','c', lower=-2.0, upper=2.0, value=1.0)
    opt_prob.addVar('x7','c', lower=-2.0, upper=2.0, value=1.0)
    opt_prob.addVar('x8','c', lower=-2.0, upper=2.0, value=1.0)
    opt_prob.addObj('f')
    print (opt_prob)
    
    # Instantiate Optimizer (SLSQP) & Solve Problem
    slsqp = SLSQP()
    slsqp.setOption('IPRINT',-1)
    slsqp(opt_prob, sens_type='FD')
    print (opt_prob.solution(0))
    
    wl_new, wu_new = airfoil_CST.getVar()
    
    
    #writefile
    fp1.write('%s\n'%nname[ii])
    fp2.write('%f %f %f %f %f %f %f %f\n'%(wl_new[0],wl_new[1],wl_new[2],wl_new[3],\
                                            wu_new[0],wu_new[1],wu_new[2],wu_new[3]))
    
    
    
    
    def plot():
        if N % 2 == 0:
            z = 0
        else:
            z = 1
        airfoil_CST2 = CST_shape(wl_new, wu_new, dz, N+z)
        coordinates = airfoil_CST2.airfoil_coor()
        x_coor = coordinates[0]
        y_coor = coordinates[1]
        ax = plt.subplot(111)
        ax.plot(x_coor, y_coor, 'g', label='CST')
        ax.plot(x1, y1, 'b', label='original')
        legend = ax.legend(loc='lower center', frameon=False)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.ylim(ymin=-0.3, ymax=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.savefig('./plot/%s.png'%nname[ii],format='png',dpi=100)
        plt.close()
    
    # UNCOMMENT TO PLOT
    plot()
    
    #print ('wl = ' , str(wl_new))
    #print ('wu = ' , str(wu_new))

fp1.close()
fp2.close()
