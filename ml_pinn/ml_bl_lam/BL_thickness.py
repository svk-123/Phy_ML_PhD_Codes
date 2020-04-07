#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 02:28:22 2019

@author: vino
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import pickle


 
##-------------
'''
d=4.91x/Sqrt(Re_x)

'''
######-BL thickness--######
########################

nu=1.0/40000.
x=np.linspace(1e-12,5,100)
Rex=x/nu
d=x/np.sqrt(Rex)
d=4.91*d

################################
###-vertical station samples

new_coord=[]
x11=np.linspace(0.5,4.5,9)
y11=np.linspace((d.max()/2)-0.005,d.max(),6)
ty=0.01
for i in range(len(x11)):
    for j in range(len(y11)):
        new_coord.append(np.asarray([x11[i],y11[j]]).transpose())
    
new_coord=np.asarray(new_coord)

plt.figure()
plt.plot(x,d)
plt.plot(new_coord[:,0],new_coord[:,1],'o')
plt.show()


