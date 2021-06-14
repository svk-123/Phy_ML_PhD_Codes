#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:38:54 2017

@author: vino
"""
# imports
from matplotlib import pyplot, cm

#plot
def plot(x,y,z,nc,name):
    pyplot.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = pyplot.tricontourf(x, y, z,nc,cmap=cm.jet)
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    pyplot.colorbar()
    pyplot.title(name)
    pyplot.xlabel('Z ')
    pyplot.ylabel('Y ')
    #pyplot.savefig(name, format='png', dpi=100)
    pyplot.show()


def plotD(x,y,z,nc,name):
    pyplot.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = pyplot.contourf(x, y, z,nc,cmap=cm.jet)
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    pyplot.colorbar()
    pyplot.title(name)
    pyplot.xlabel('Z ')
    pyplot.ylabel('Y ')
    #pyplot.savefig(name, format='png', dpi=100)
    pyplot.show()
