#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:50:55 2020

@author: vino
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os import listdir
from os.path import isfile, join




''''
u-cfd1 y1 u-pinn1 y1 u-cfd2 y2 u-pinn2 y2  ....
'''

'''
###########_---- Wos - BC-1,2,3###############################
suff='wos_re100'
uy1=np.loadtxt('./paper_files/uv/u_re100_dp_dv.dat',skiprows=1)
uy2=np.loadtxt('./paper_files/uv/u_re100_nodp_nodv.dat',skiprows=1)
uy3=np.loadtxt('./paper_files/uv/u_re100_4s_puv.dat',skiprows=1)

plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(uy1[:,0],uy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,2],uy1[:,3],'g',linewidth=3,label='PINN-BC-1')
plt.plot(uy2[:,2],uy2[:,3],'b',linewidth=3,label='PINN-BC-2')
plt.plot(uy3[:,2],uy3[:,3],'r',linewidth=3,label='PINN-BC-3')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(uy1[:,4],uy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,6],uy1[:,7],'g',linewidth=3,label='PINN-BC-1')
plt.plot(uy2[:,6],uy2[:,7],'b',linewidth=3,label='PINN-BC-2')
plt.plot(uy3[:,6],uy3[:,7],'r',linewidth=3,label='PINN-BC-3')
plt.xlabel('u/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(uy1[:,8],uy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,10],uy1[:,11],'g',linewidth=3,label='PINN-BC-1')
plt.plot(uy2[:,10],uy2[:,11],'b',linewidth=3,label='PINN-BC-2')
plt.plot(uy3[:,10],uy3[:,11],'r',linewidth=3,label='PINN-BC-3')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/u_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()


####################################################################################


vy1=np.loadtxt('./paper_files/uv/v_re100_dp_dv.dat',skiprows=1)
vy2=np.loadtxt('./paper_files/uv/v_re100_nodp_nodv.dat',skiprows=1)
vy3=np.loadtxt('./paper_files/uv/v_re100_4s_puv.dat',skiprows=1)

plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(vy1[:,0],vy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,2],vy1[:,3],'g',linewidth=3,label='PINN-BC-1')
plt.plot(vy2[:,2],vy2[:,3],'b',linewidth=3,label='PINN-BC-2')
plt.plot(vy3[:,2],vy3[:,3],'r',linewidth=3,label='PINN-BC-3')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(vy1[:,4],vy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,6],vy1[:,7],'g',linewidth=3,label='PINN-BC-1')
plt.plot(vy2[:,6],vy2[:,7],'b',linewidth=3,label='PINN-BC-2')
plt.plot(vy3[:,6],vy3[:,7],'r',linewidth=3,label='PINN-BC-3')
plt.xlabel('v/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(vy1[:,8],vy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,10],vy1[:,11],'g',linewidth=3,label='PINN-BC-1')
plt.plot(vy2[:,10],vy2[:,11],'b',linewidth=3,label='PINN-BC-2')
plt.plot(vy3[:,10],vy3[:,11],'r',linewidth=3,label='PINN-BC-3')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/v_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()
'''

'''
###########_---- Re100--Ws - S1-S2###############################
suff='ws_re100'
uy1=np.loadtxt('./paper_files/uv/u_re100_nodp_nodv_ws_x8_away.dat',skiprows=1)
uy2=np.loadtxt('./paper_files/uv/u_re100_nodp_nodv_ws_x8.dat',skiprows=1)


plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(uy1[:,0],uy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,2],uy1[:,3],'g',linewidth=3,label='PINN-S-1')
plt.plot(uy2[:,2],uy2[:,3],'b',linewidth=3,label='PINN-S-2')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(uy1[:,4],uy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,6],uy1[:,7],'g',linewidth=3,label='PINN-S-1')
plt.plot(uy2[:,6],uy2[:,7],'b',linewidth=3,label='PINN-S-2')
plt.xlabel('u/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(uy1[:,8],uy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,10],uy1[:,11],'g',linewidth=3,label='PINN-S-1')
plt.plot(uy2[:,10],uy2[:,11],'b',linewidth=3,label='PINN-S-2')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/u_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()


####################################################################################

vy1=np.loadtxt('./paper_files/uv/v_re100_nodp_nodv_ws_x8_away.dat',skiprows=1)
vy2=np.loadtxt('./paper_files/uv/v_re100_nodp_nodv_ws_x8.dat',skiprows=1)

plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(vy1[:,0],vy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,2],vy1[:,3],'g',linewidth=3,label='PINN-S-1')
plt.plot(vy2[:,2],vy2[:,3],'b',linewidth=3,label='PINN-S-2')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(vy1[:,4],vy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,6],vy1[:,7],'g',linewidth=3,label='PINN-S-1')
plt.plot(vy2[:,6],vy2[:,7],'b',linewidth=3,label='PINN-S-2')
plt.xlabel('v/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(vy1[:,8],vy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,10],vy1[:,11],'g',linewidth=3,label='PINN-S-1')
plt.plot(vy2[:,10],vy2[:,11],'b',linewidth=3,label='PINN-S-2')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/v_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()
'''




'''

###########_---- Re100--wos-PINN-NN###############################
suff='pinn_nn_re100'
uy1=np.loadtxt('./paper_files/uv/u_re100_nodp_nodv.dat',skiprows=1)
uy2=np.loadtxt('./paper_files/uv/u_re100_nodp_nodv_nopinn.dat',skiprows=1)


plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(uy1[:,0],uy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,2],uy1[:,3],'g',linewidth=3,label='PINN')
plt.plot(uy2[:,2],uy2[:,3],'b',linewidth=3,label='NN')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(uy1[:,4],uy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,6],uy1[:,7],'g',linewidth=3,label='PINN')
plt.plot(uy2[:,6],uy2[:,7],'b',linewidth=3,label='NN')
plt.xlabel('u/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(uy1[:,8],uy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,10],uy1[:,11],'g',linewidth=3,label='PINN')
plt.plot(uy2[:,10],uy2[:,11],'b',linewidth=3,label='NN')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/u_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()


####################################################################################

vy1=np.loadtxt('./paper_files/uv/v_re100_nodp_nodv.dat',skiprows=1)
vy2=np.loadtxt('./paper_files/uv/v_re100_nodp_nodv_nopinn.dat',skiprows=1)

plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(vy1[:,0],vy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,2],vy1[:,3],'g',linewidth=3,label='PINN')
plt.plot(vy2[:,2],vy2[:,3],'b',linewidth=3,label='NN')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(vy1[:,4],vy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,6],vy1[:,7],'g',linewidth=3,label='PINN')
plt.plot(vy2[:,6],vy2[:,7],'b',linewidth=3,label='NN')
plt.xlabel('v/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(vy1[:,8],vy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,10],vy1[:,11],'g',linewidth=3,label='PINN')
plt.plot(vy2[:,10],vy2[:,11],'b',linewidth=3,label='NN')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/v_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()

'''














'''

##########-- high Re--------------------------------------

suff='ws_re20000'
uy1=np.loadtxt('./paper_files/uv/u_re20000_nodp_nodv_ws_x8.dat',skiprows=1)


plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(uy1[:,0],uy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,2],uy1[:,3],'g',linewidth=3,label='PINN')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(uy1[:,4],uy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,6],uy1[:,7],'g',linewidth=3,label='PINN')
plt.xlabel('u/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(uy1[:,8],uy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,10],uy1[:,11],'g',linewidth=3,label='PINN')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/u_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()


####################################################################################

vy1=np.loadtxt('./paper_files/uv/v_re20000_nodp_nodv_ws_x8.dat',skiprows=1)


plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(vy1[:,0],vy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,2],vy1[:,3],'g',linewidth=3,label='PINN')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(vy1[:,4],vy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,6],vy1[:,7],'g',linewidth=3,label='PINN')
plt.xlabel('v/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(vy1[:,8],vy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,10],vy1[:,11],'g',linewidth=3,label='PINN')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/v_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()
'''






#paper plot uv for review
#plot data with noise

'''
u-cfd1 y1 u-pinn1 y1 u-cfd2 y2 u-pinn2 y2  ....
'''

'''
###########_---- Wos - BC-1,2,3###############################
suff='uv_noise'
uy1=np.loadtxt('./paper_files/uv/u_re100_nodp_nodv_ws_x8.dat',skiprows=1)
uy2=np.loadtxt('./paper_files/uv_review/u_re100_5p.dat',skiprows=1)
uy3=np.loadtxt('./paper_files/uv_review/u_re100_10p.dat',skiprows=1)

plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(uy1[:,0],uy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,2],uy1[:,3],'g',linewidth=3,label='PINN')
plt.plot(uy2[:,2],uy2[:,3],'b',linewidth=3,label='PINN-noise 5%')
plt.plot(uy3[:,2],uy3[:,3],'r',linewidth=3,label='PINN-noise 10%')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(uy1[:,4],uy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,6],uy1[:,7],'g',linewidth=3,label='PINN-BC-1')
plt.plot(uy2[:,6],uy2[:,7],'b',linewidth=3,label='PINN-BC-2')
plt.plot(uy3[:,6],uy3[:,7],'r',linewidth=3,label='PINN-BC-3')
plt.xlabel('u/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(uy1[:,8],uy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,10],uy1[:,11],'g',linewidth=3,label='PINN-')
plt.plot(uy2[:,10],uy2[:,11],'b',linewidth=3,label='PINN-noise 5%')
plt.plot(uy3[:,10],uy3[:,11],'r',linewidth=3,label='PINN-noise 10%')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/u_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()


####################################################################################


vy1=np.loadtxt('./paper_files/uv/v_re100_nodp_nodv_ws_x8.dat',skiprows=1)
vy2=np.loadtxt('./paper_files/uv_review/v_re100_5p.dat',skiprows=1)
vy3=np.loadtxt('./paper_files/uv_review/v_re100_10p.dat',skiprows=1)

plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(vy1[:,0],vy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,2],vy1[:,3],'g',linewidth=3,label='PINN-BC-1')
plt.plot(vy2[:,2],vy2[:,3],'b',linewidth=3,label='PINN-BC-2')
plt.plot(vy3[:,2],vy3[:,3],'r',linewidth=3,label='PINN-BC-3')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(vy1[:,4],vy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,6],vy1[:,7],'g',linewidth=3,label='PINN-BC-1')
plt.plot(vy2[:,6],vy2[:,7],'b',linewidth=3,label='PINN-BC-2')
plt.plot(vy3[:,6],vy3[:,7],'r',linewidth=3,label='PINN-BC-3')
plt.xlabel('v/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(vy1[:,8],vy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,10],vy1[:,11],'g',linewidth=3,label='PINN')
plt.plot(vy2[:,10],vy2[:,11],'b',linewidth=3,label='PINN-noise 5%')
plt.plot(vy3[:,10],vy3[:,11],'r',linewidth=3,label='PINN-noise 10%')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/v_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()
'''



#paper plot uv for review
#plot data with noise

'''
u-cfd1 y1 u-pinn1 y1 u-cfd2 y2 u-pinn2 y2  ....
'''


###########_---- Wos - BC-1,2,3###############################
suff='uv_sample_loc'
uy1=np.loadtxt('./paper_files/uv/u_re100_nodp_nodv_ws_x8.dat',skiprows=1)
uy2=np.loadtxt('./paper_files/uv_review/u_re100_sample_p8.dat',skiprows=1)
uy3=np.loadtxt('./paper_files/uv_review/u_re100_sample_p10.dat',skiprows=1)

plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(uy1[:,0],uy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,2],uy1[:,3],'g',linewidth=3,label='PINN')
plt.plot(uy2[:,2],uy2[:,3],'b',linewidth=3,label='PINN-noise 5%')
plt.plot(uy3[:,2],uy3[:,3],'r',linewidth=3,label='PINN-noise 10%')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(uy1[:,4],uy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,6],uy1[:,7],'g',linewidth=3,label='PINN-BC-1')
plt.plot(uy2[:,6],uy2[:,7],'b',linewidth=3,label='PINN-BC-2')
plt.plot(uy3[:,6],uy3[:,7],'r',linewidth=3,label='PINN-BC-3')
plt.xlabel('u/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(uy1[:,8],uy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(uy1[:,10],uy1[:,11],'g',linewidth=3,label='PINN')
plt.plot(uy2[:,10],uy2[:,11],'b',linewidth=3,label='PINN-sample @ 20%')
plt.plot(uy3[:,10],uy3[:,11],'r',linewidth=3,label='PINN-sample @ 0%')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(a)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/u_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()


####################################################################################


vy1=np.loadtxt('./paper_files/uv/v_re100_nodp_nodv_ws_x8.dat',skiprows=1)
vy2=np.loadtxt('./paper_files/uv_review/v_re100_sample_p8.dat',skiprows=1)
vy3=np.loadtxt('./paper_files/uv_review/v_re100_sample_p10.dat',skiprows=1)

plt.figure(figsize=(6, 4), dpi=100)
mei=2
plt.subplot(1,3,1)
plt.plot(vy1[:,0],vy1[:,1],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,2],vy1[:,3],'g',linewidth=3,label='PINN-BC-1')
plt.plot(vy2[:,2],vy2[:,3],'b',linewidth=3,label='PINN-BC-2')
plt.plot(vy3[:,2],vy3[:,3],'r',linewidth=3,label='PINN-BC-3')
#plt.plot(u3a,ya,'r',linewidth=3,label='NN')
#plt.legend(fontsize=14)
plt.ylabel('y/$\delta$(x)',fontsize=20)
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(vy1[:,4],vy1[:,5],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,6],vy1[:,7],'g',linewidth=3,label='PINN-BC-1')
plt.plot(vy2[:,6],vy2[:,7],'b',linewidth=3,label='PINN-BC-2')
plt.plot(vy3[:,6],vy3[:,7],'r',linewidth=3,label='PINN-BC-3')
plt.xlabel('v/U$_\inf$',fontsize=20)
plt.yticks([])
#plt.xlim(-0.1,1.2)
#plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(vy1[:,8],vy1[:,9],'o',mfc='None',mew=1.5,mec='k',ms=10,markevery=mei,label='CFD')
plt.plot(vy1[:,10],vy1[:,11],'g',linewidth=3,label='PINN')
plt.plot(vy2[:,10],vy2[:,11],'b',linewidth=3,label='PINN-sample @ 20%')
plt.plot(vy3[:,10],vy3[:,11],'r',linewidth=3,label='PINN-sample @ 0%')
plt.legend(loc="upper left", bbox_to_anchor=[1, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
plt.yticks([])
#plt.xlim(-0.5,1.2)
#plt.ylim(-0.05,1.05)

plt.figtext(0.4, 0.00, '(b)', wrap=True, horizontalalignment='center', fontsize=24)
plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.9, left = 0.0, hspace = 0.0, wspace = 0.1)
plt.savefig('./plot/v_%s.tiff'%suff, format='tiff', bbox_inches='tight',dpi=300)
plt.show()   
plt.close()
