__author__ = 'ryanbarr'

import numpy as np
import matplotlib.pylab as plt
from kulfan_to_coord import CST_shape
from pyOpt import Optimization, SLSQP
import os, sys, time
import pdb
from os import listdir
from os.path import isfile, isdir, join
from numpy import linalg as LA

#load file name
file = open("./cst_parameter/foilname.dat")
line = file.read().replace("\n", " ")
file.close()
tmp=line.split(" ")
nname=np.asarray(tmp[:1433])

#cst para
dz=0
N=100

cst_para=np.loadtxt("./cst_parameter/cst.dat")


train_l2=[]
for ii in range(1433):
    
    print(ii)
    
    foil=np.loadtxt('./picked_uiuc_101/%s.dat'%nname[ii])
    
    x1 = foil[:,0]
    y1 = foil[:,1]

    if N % 2 == 0:
        z = 0
    else:
        z = 1
    airfoil_CST2 = CST_shape(cst_para[ii][0:4], cst_para[ii][4:8], dz, N+z)
    coordinates = airfoil_CST2.airfoil_coor()
    x_coor = coordinates[0]
    y_coor = coordinates[1]
#    ax = plt.subplot(111)
#    ax.plot(x_coor, y_coor, 'g', label='CST')
#    ax.plot(x1, y1, 'b', label='original')
#    legend = ax.legend(loc='lower center', frameon=False)
#    plt.xlabel('x/c')
#    plt.ylabel('y/c')
#    plt.ylim(ymin=-0.25, ymax=0.25)
#    ax.spines['right'].set_visible(False)
#    ax.spines['top'].set_visible(False)
#    ax.yaxis.set_ticks_position('left')
#    ax.xaxis.set_ticks_position('bottom')
#    plt.savefig('./plot/%s.png'%nname[ii],format='png',dpi=100)
#    plt.show()
#    plt.close()


    #calculate error norm
    tmp=y_coor-y1[:100]
    train_l2.append( (LA.norm(tmp)/LA.norm(y1[:100]))*100 )
    
    
print ("train_l2_avg",sum(train_l2)/len(train_l2))

#error plot
plt.figure(figsize=(6,5),dpi=100)
plt.hist(train_l2, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
plt.ylabel('Number of Samples',fontsize=20)
plt.xlabel('$L_2$ relative error(%)',fontsize=20)
plt.figtext(0.40, 0.01, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
#plt.xlim([-0.0,0.25])
#plt.xticks([0,0.5,1.])
plt.savefig('tr_tot.tiff',format='tiff', bbox_inches='tight',dpi=300)
plt.show()