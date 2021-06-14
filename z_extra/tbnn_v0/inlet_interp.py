#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:31:27 2017

@author: vino
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 19:35:45 2017

@author: vino
"""

# imports
import os
import glob

import numpy as np
from matplotlib import pyplot, cm
import pandas as pd
from scipy import interpolate

#time 
import time
start_time = time.time()

# read DNS data
dataframe = pd.read_csv('z.txt', sep=',', header=None, skiprows=0)
dataset = dataframe.values
z=np.asarray(dataset)

dataframe = pd.read_csv('y.txt', sep=',', header=None, skiprows=0)
dataset = dataframe.values
y=np.asarray(dataset)

dataframe = pd.read_csv('um.txt', sep=',', header=None, skiprows=0)
dataset = dataframe.values
um=np.asarray(dataset)

# interpolate
um=um/um.max()

Z, Y = np.meshgrid(z, y, copy=False)

f = interpolate.interp2d(Z, Y, um, kind='linear')


#read mesh data
dataframe = pd.read_csv('inletCoord.dat', sep='\t', header=None, skiprows=0)
dataset = dataframe.values
ip=np.asarray(dataset[:,0])
xi=np.asarray(dataset[:,1])
yi=np.asarray(dataset[:,2])
zi=np.asarray(dataset[:,3])

#iterpolate mesh points
umo=np.zeros((len(ip)))
for i in range(len(ip)):
    umo[i]=f(zi[i],yi[i])


#plot
def plot(pp,name,nc):
    pyplot.figure(figsize=(6, 5), dpi=100)
    cp = pyplot.tricontour(zi, yi, pp,nc)
    pyplot.clabel(cp, inline=True,fontsize=8)
    pyplot.colorbar()
    pyplot.title('Contour Plot')
    pyplot.xlabel('X ')
    pyplot.ylabel('Y ')
    #pyplot.savefig(name, format='png', dpi=100)
    pyplot.show()

plot(umo,'na',100)

fp= open("velOut.dat","w+")
for i in range(len(ip)):
    fp.write("%d\t %f\t %f\t %f\t %f\t%f\n" %(ip[i],xi[i],yi[i],zi[i],umo[i],umo[i]))
fp.close()    

#average velocity
ya=np.linspace(-1,1,100)
za=np.linspace(-1,1,100)
uma=np.zeros((len(ya)))
for i in range(len(ya)):
    uma[i]=f(za[i],ya[i])

ub=np.average(uma)
print ("ub= %f\t ub*1.243= %f\n" %(ub,ub*1.243))
