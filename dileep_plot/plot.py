# imports
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import interpolate
from os import listdir
from os.path import isfile,isdir, join






path='./24_Feb_2021/'
plot_path='./plot_24_Feb_2021/'
tmp=[f for f in listdir(path) if isfile(join(path, f))]


for ii in range(8):
    print(tmp[ii])
    with open(path+'%s'%tmp[ii], 'r') as infile:
        x=[]
        a12=[]
        a34=[]
        a56=[]
        a78=[]
        data0=infile.readlines()
        count=0
        for line in data0[3:len(data0)]:
    
            #tmp=line.replace('\t\t\t\t\t\t','') 
            #tmp=line.replace('\t\t\t\t','')
            #tmp=line.replace('\t\t','')
            a=line.split('\t')        
            a=np.asarray(a)
            #print(a)
            #print(count)
            
            a12.append(a[0:2])
            
            if(a[3]!=''):
                a34.append(a[2:4])
                
            if(a[5]!=''):        
                a56.append(a[4:6])
            
            if(a[7]!=''):
                a78.append(a[6:8])
            
    
                
                
    a12=np.asarray(a12)    
    a34=np.asarray(a34) 
    a56=np.asarray(a56) 
    a78=np.asarray(a78) 
            
    a12=a12.astype(np.float)        
    a34=a34.astype(np.float)   
    a56=a56.astype(np.float)   
    a78=a78.astype(np.float)          
    
    
    xin=150
    xout=500
    cut=-550
    
    for k in range(len(a56)):
        if(a56[k,0] > xin and a56[k,0] < xout and a56[k,1] > cut):
            a56[k,1]=cut
    
    # xin=400
    # xout=500
    # cut=-400
    
    # for k in range(len(a56)):
    #     if(a56[k,0] > xin and a56[k,0] < xout and a56[k,1] > cut):
    #         a56[k,1]=cut
            
    
    #https://matplotlib.org/examples/color/named_colors.html
    c=['g','navy','r','k','r','m','darkorange','lime','pink','purple','peru','gold','olive','salmon','brown'] 
    
    L=50000
    plt.figure(figsize=(6,5),dpi=100)
    for i in range(1):
        #total loss
        plt.plot(a56[:,0], a56[:,1]+720,'%s'%c[1],marker='None',mfc='r',ms=12,lw=1.25,markevery=1,label='UV_215')
        plt.xlabel('Volume(ml)',fontsize=16)
        plt.ylabel('mAU',fontsize=16)
        plt.legend(loc="upper left", bbox_to_anchor=[0.4, 1], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
        plt.twinx()
        plt.plot(a78[:,0], a78[:,1],'%s'%c[2],marker='None',mfc='r',ms=12,lw=1.25,markevery=1,label='%B')
        
    plt.legend(loc="upper left", bbox_to_anchor=[0.4, 0.9], ncol=1, fontsize=14, frameon=False, shadow=False, fancybox=False,title='')
    #plt.xlabel('Volume(ml)',fontsize=20)
    plt.ylabel('%B',fontsize=16)
    
    #plt.yscale('log')
    #plt.figtext(0.45, 0.03, '(a)', wrap=True, horizontalalignment='center', fontsize=24)    
    plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
    #plt.xticks(range(0,2001,500))
    #plt.xlim([-50,6000])
    #plt.ylim([5e-6,1e-3])    
    plt.savefig(plot_path+'%s.png'%tmp[ii], format='png', bbox_inches='tight',dpi=300)
    plt.show()
        