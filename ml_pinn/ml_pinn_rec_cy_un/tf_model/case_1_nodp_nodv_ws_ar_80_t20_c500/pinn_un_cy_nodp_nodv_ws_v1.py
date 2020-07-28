#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:35:28 2019

@author: vino
"""

"""
@author: originnaly written by Maziar Raissi
Further modified by Vinothkumar S.

ws indicates with sampling @y=0.05 or 0.02

"""
'''

'''

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import pickle

start_time = time.time()

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, t, p, u, v, \
                 x_iw, y_iw, t_iw, p_iw, u_iw, v_iw, nx_iw, ny_iw, \
                 x_oo, y_oo, t_oo, p_oo, u_oo, v_oo, nx_oo, ny_oo, \
                              xg, yg, tg):
                  
        self.x = x
        self.y = y
        self.t = t
        
        self.u = u
        self.v = v
        self.p = p

        self.x_iw = x_iw
        self.y_iw = y_iw
        self.t_iw = t_iw      
        
        self.u_iw = u_iw
        self.v_iw = v_iw
        self.nx_iw= nx_iw
        self.ny_iw= ny_iw
        
        self.x_oo = x_oo
        self.y_oo = y_oo
        self.t_oo = t_oo
        
        self.p_oo = p_oo
        self.nx_oo= nx_oo
        self.ny_oo= ny_oo
                  
        self.xg = xg
        self.yg = yg
        self.tg = tg
                
        # Initialize parameters (1/140)
        self.nu = tf.constant([1./100.], dtype=tf.float32)
        
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]],name='input0')
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]],name='input1')
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]],name='input2')
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        self.p_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.x_iw_tf = tf.placeholder(tf.float32, shape=[None, self.x_iw.shape[1]],name='input1a')
        self.y_iw_tf = tf.placeholder(tf.float32, shape=[None, self.y_iw.shape[1]],name='input1b')
        self.t_iw_tf = tf.placeholder(tf.float32, shape=[None, self.y_iw.shape[1]],name='input1c')
        
        self.u_iw_tf = tf.placeholder(tf.float32, shape=[None, self.u_iw.shape[1]])
        self.v_iw_tf = tf.placeholder(tf.float32, shape=[None, self.v_iw.shape[1]]) 
        self.nx_iw_tf = tf.placeholder(tf.float32, shape=[None, self.nx_iw.shape[1]],name='input1d')         
        self.ny_iw_tf = tf.placeholder(tf.float32, shape=[None, self.ny_iw.shape[1]],name='input1e')   
        
        self.x_oo_tf = tf.placeholder(tf.float32, shape=[None, self.x_oo.shape[1]])
        self.y_oo_tf = tf.placeholder(tf.float32, shape=[None, self.y_oo.shape[1]])
        self.t_oo_tf = tf.placeholder(tf.float32, shape=[None, self.t_oo.shape[1]])
        
        self.p_oo_tf = tf.placeholder(tf.float32, shape=[None, self.p_oo.shape[1]])
        self.nx_oo_tf = tf.placeholder(tf.float32, shape=[None, self.nx_oo.shape[1]])         
        self.ny_oo_tf = tf.placeholder(tf.float32, shape=[None, self.ny_oo.shape[1]])   
          
        self.xg_tf = tf.placeholder(tf.float32, shape=[None, self.xg.shape[1]])
        self.yg_tf = tf.placeholder(tf.float32, shape=[None, self.yg.shape[1]])
        self.tg_tf = tf.placeholder(tf.float32, shape=[None, self.tg.shape[1]])        

        
        #MSE wall
        self.u_iw_pred, self.v_iw_pred, _, _= self.net_NS1(self.x_iw_tf, self.y_iw_tf, self.t_iw_tf, self.nx_iw_tf, self.ny_iw_tf)
        
        #MSE outlet
        _, _, self.p_oo_pred, _ = self.net_NS2(self.x_oo_tf, self.y_oo_tf, self.t_oo_tf, self.nx_oo_tf, self.ny_oo_tf) 
        
        #gov Eqn
        self.f_c_pred, self.f_u_pred, self.f_v_pred = self.net_NS3(self.xg_tf, self.yg_tf, self.tg_tf)
        self.u_pred, self.v_pred, self.p_pred  = self.net_NS0(self.x_tf, self.y_tf, self.t_tf)
        

        self.loss_1 = tf.reduce_mean(tf.square(self.u_iw_tf - self.u_iw_pred)) + \
                        tf.reduce_mean(tf.square(self.v_iw_tf - self.v_iw_pred))
                                                            
        self.loss_2 = tf.reduce_mean(tf.square(self.p_oo_tf - self.p_oo_pred)) 
        
        self.loss_3 = tf.reduce_mean(tf.square(self.f_c_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred))
                    
        self.loss_0 = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_mean(tf.square(self.p_tf - self.p_pred))   
                    
        self.loss = self.loss_0 + self.loss_1 + self.loss_2 + self.loss_3
                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 100,
                                                                           'maxls': 100,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.train_op_Adam = tf.train.AdamOptimizer(self.tf_lr).minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
    
    def neural_net(self, X):

        #create model
        l1 = tf.layers.dense(X,  100, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 100, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 100, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 100, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 100, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 100, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 100, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 100, activation=tf.nn.tanh)
        Y  = tf.layers.dense(l1,3,activation=None,name='prediction')
                
        return Y
    

        
    def net_NS1(self, x, y, t, nx, ny):
        
        with tf.variable_scope("NS1"):
            uvp = self.neural_net(tf.concat([x,y,t], 1))
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
        
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]     
        pn  = p_x*nx + p_y*ny
        
        return u, v, p, pn

    def net_NS2(self, x, y, t, nx, ny):
        
        with tf.variable_scope("NS1",reuse=True):
            uvp = self.neural_net(tf.concat([x,y,t], 1))
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
      
  
        
        return u, v, p, p

   
    def net_NS3(self, x, y, t):
        
        with tf.variable_scope("NS1",reuse=True):
            uvp = self.neural_net(tf.concat([x,y,t], 1))
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
      
        u_t = tf.gradients(u, t)[0]        
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        
        v_t = tf.gradients(v, t)[0]        
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_c =  u_x + v_y
        f_u =  u_t + (u*u_x + v*u_y) + p_x - (self.nu)*(u_xx + u_yy) 
        f_v =  v_t + (u*v_x + v*v_y) + p_y - (self.nu)*(v_xx + v_yy)
        
        return  f_c, f_u, f_v
    
    def net_NS0(self, x, y, t):
        
        with tf.variable_scope("NS1",reuse=True):
            uvp = self.neural_net(tf.concat([x,y,t], 1))
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
        
        
        return u, v, p
    
    def callback(self, loss, loss_1, loss_2, loss_3):
        print('Loss: %.6e %.6e %.6e %.6e \n' % (loss,loss_1,loss_2, loss_3))       
        self.fp.write('00, %.6e, %.6e, %.6e %.6e \n'% (loss,loss_1,loss_2, loss_3)) 
        
      
    def train(self, nIter, lbfgs=False): 
        
        self.fp=open('./tf_model/conv.dat','w')
    
    

        total_batch= 1.0

        
        lr=0.001
        min_lr=1e-7
        #reduce lr iter(patience)
        rli=3000
        #numbers to avg
        L=30
        #lr eps
        l_eps=1e-7
        
        #early stop wait
        estop=3000
        e_eps=1e-7
        
        start_time = time.time()
        
        my_hist=[]
        
        #epochs traings
        self.fp.write('Iter, Loss, Loss-MSE, Loss-Res, LR, Time \n')
        count=0
        while(count < nIter):
            count=count+1
            avg_loss = 0.
            avg_lv_1 = 0.
            avg_lv_2 = 0.
            avg_lv_3 = 0.
            
            #batch training
            for i in range(1):
                            
                
                tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t, self.u_tf: self.u, self.v_tf: self.v, self.p_tf: self.p, \
                           self.x_iw_tf: self.x_iw, self.y_iw_tf: self.y_iw, self.t_iw_tf: self.t_iw, self.u_iw_tf: self.u_iw, self.v_iw_tf: self.v_iw, \
                           self.x_oo_tf: self.x_oo, self.y_oo_tf: self.y_oo, self.t_oo_tf: self.t_oo, self.p_oo_tf: self.p_oo,\
                           self.tf_lr:lr, self.xg_tf: self.xg, self.yg_tf: self.yg,self.tg_tf: self.tg}
            
                _,loss_value,lv_1,lv_2,lv_3=self.sess.run([self.train_op_Adam,self.loss,self.loss_1,self.loss_2,self.loss_3], tf_dict)
                avg_loss += loss_value / total_batch
                avg_lv_1 += lv_1 / total_batch
                avg_lv_2 += lv_2 / total_batch   
                avg_lv_3 += lv_3 / total_batch 
                
            my_hist.append(avg_loss)
            
            #reduce lr
            if(len(my_hist) > rli  and lr > min_lr):
                if ((sum(my_hist[-rli:-rli+L]) - sum(my_hist[-L-1:-1])) < (l_eps*L) ):
                    lr=lr*0.33
                    print('Reduce Learning rate',lr,len(my_hist[-L-1:-1]),len(my_hist[-rli:-rli+L]))
                    my_hist=[]
                    
                    self.fp.write('Reduce Learning rate: %f \n' %lr)
                        
            #early stop        
            if(len(my_hist) > estop  and lr <= min_lr):
                if ( (sum(my_hist[-estop:-estop+L]) - sum(my_hist[-L-1:-1])) < (e_eps*L) ):
                    print ('Early STOP STOP STOP')
                    self.fp.write('Early STOP STOP STOP')
                    nIter=count
                    
            #print
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.6e, Loss-1:%0.6e, Loss-2:%0.6e, Loss-2:%0.6e, lr:%0.6f, Time: %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2, avg_lv_3, lr, elapsed))
            
            self.fp.write('%d, %.6e, %0.6e, %0.6e, %0.6e, %0.6e, %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2, avg_lv_3, lr, elapsed))    
            start_time = time.time()
            
            #save model
            if ((count % 10000) ==0):
                model.save_model(count)
                
       
        #final_optimization using lbfgsb
        if (lbfgs==True):
                    
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t, self.u_tf: self.u, self.v_tf: self.v, self.p_tf: self.p, \
                           self.x_iw_tf: self.x_iw, self.y_iw_tf: self.y_iw, self.t_iw_tf: self.t_iw, self.u_iw_tf: self.u_iw, self.v_iw_tf: self.v_iw, \
                           self.x_oo_tf: self.x_oo, self.y_oo_tf: self.y_oo, self.t_oo_tf: self.t_oo, self.p_oo_tf: self.p_oo,\
                           self.tf_lr:lr, self.xg_tf: self.xg, self.yg_tf: self.yg,self.tg_tf: self.tg}
                
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss,self.loss_1,self.loss_2,self.loss_3],
                                    loss_callback = self.callback)
 

        self.fp.close()
            
    def predict(self, x_star, y_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, v_star, p_star    

    def save_model(self,count):
        self.saver.save(self.sess,'./tf_model/model_%d'%count)        
        
if __name__ == "__main__": 
           
    ######################################################################
    ######################## MSE Data ####################################
    ######################################################################
    
    path='../data_file/'  
    #import wall bc
    #x,y,p,u,v
    xyu_inlet=np.loadtxt(path + 'cy_inlet3_3222.dat',skiprows=1)
    
    x_inlet = xyu_inlet[:,0:1]
    y_inlet = xyu_inlet[:,1:2]
    t_inlet = xyu_inlet[:,2:3]
    p_inlet = xyu_inlet[:,3:4]
    u_inlet = xyu_inlet[:,4:5]
    v_inlet = xyu_inlet[:,5:6]
    nx_inlet= xyu_inlet[:,5:6]
    ny_inlet= xyu_inlet[:,5:6]

    xyu_wall=np.loadtxt(path + 'cy_wall_bc_80_t.dat',skiprows=1)
    
    x_wall = xyu_wall[:,0:1]
    y_wall = xyu_wall[:,1:2]
    t_wall = xyu_wall[:,2:3]    
    p_wall = xyu_wall[:,3:4]
    u_wall = xyu_wall[:,4:5]
    v_wall = xyu_wall[:,5:6]   
    nx_wall= xyu_wall[:,5:6]
    ny_wall= xyu_wall[:,5:6]

    
    xyu_outlet_t=np.loadtxt(path + 'cy_outlet1_3222.dat',skiprows=1)
    
    x_outlet_t = xyu_outlet_t[:,0:1]
    y_outlet_t = xyu_outlet_t[:,1:2]
    t_outlet_t = xyu_outlet_t[:,2:3]    
    p_outlet_t = xyu_outlet_t[:,3:4]
    u_outlet_t = xyu_outlet_t[:,4:5]
    v_outlet_t = xyu_outlet_t[:,5:6]       
    nx_outlet_t= xyu_outlet_t[:,5:6]
    ny_outlet_t= xyu_outlet_t[:,5:6]
    
    #inlet-wall
    x_iw  = np.concatenate((xyu_inlet[:,0:1], xyu_wall[:,0:1]),axis=0)
    y_iw  = np.concatenate((xyu_inlet[:,1:2], xyu_wall[:,1:2]),axis=0)
    t_iw  = np.concatenate((xyu_inlet[:,2:3], xyu_wall[:,2:3]),axis=0)
    p_iw  = np.concatenate((xyu_inlet[:,3:4], xyu_wall[:,3:4]),axis=0)
    u_iw  = np.concatenate((xyu_inlet[:,4:5], xyu_wall[:,4:5]),axis=0)    
    v_iw  = np.concatenate((xyu_inlet[:,5:6], xyu_wall[:,5:6]),axis=0) 
    nx_iw =  v_iw
    ny_iw =  v_iw  

    #outlet-outlet
    x_oo  = xyu_outlet_t[:,0:1]
    y_oo  = xyu_outlet_t[:,1:2]
    t_oo  = xyu_outlet_t[:,2:3]
    p_oo  = xyu_outlet_t[:,3:4]
    u_oo  = xyu_outlet_t[:,4:5]   
    v_oo  = xyu_outlet_t[:,5:6]
    nx_oo = xyu_outlet_t[:,5:6]   
    ny_oo = xyu_outlet_t[:,5:6]

    #sampling
    xyu_s=np.loadtxt(path + 'cy_sample_ar_10x2.dat',skiprows=1)
    
    idx = np.random.choice(len(xyu_s), len(xyu_s), replace=False)
    x_s = xyu_s[idx,0:1]
    y_s = xyu_s[idx,1:2]
    t_s = xyu_s[idx,2:3]
    p_s = xyu_s[idx,3:4]
    u_s = xyu_s[idx,4:5]
    v_s = xyu_s[idx,5:6] 
    
    x_train = xyu_s[idx,0:1]
    y_train = xyu_s[idx,1:2]
    t_train = xyu_s[idx,2:3]    
    p_train = xyu_s[idx,3:4]
    u_train = xyu_s[idx,4:5]
    v_train = xyu_s[idx,5:6]     
    
#    # MSE points
#    x_train = np.concatenate((xyu_inlet[:,0:1],xyu_outlet_t[:,0:1],xyu_outlet_r[:,0:1],xyu_s[:,0:1]),axis=0)
#    y_train = np.concatenate((xyu_inlet[:,1:2],xyu_outlet_t[:,1:2],xyu_outlet_r[:,1:2],xyu_s[:,1:2]),axis=0)
#    p_train = np.concatenate((xyu_inlet[:,2:3],xyu_outlet_t[:,2:3],xyu_outlet_r[:,2:3],xyu_s[:,2:3]),axis=0)
#    u_train = np.concatenate((xyu_inlet[:,3:4],xyu_outlet_t[:,3:4],xyu_outlet_r[:,3:4],xyu_s[:,3:4]),axis=0)    
#    v_train = np.concatenate((xyu_inlet[:,4:5],xyu_outlet_t[:,4:5],xyu_outlet_r[:,4:5],xyu_s[:,4:5]),axis=0)  
    
    ######################################################################
    ######################## Gov Data ####################################
    ######################################################################    
    
    xyu_int=np.loadtxt(path + 'cy_internal_centers_500_3222.dat',skiprows=1)    
    
    N_train=10000
    idx = np.random.choice(len(xyu_int), N_train, replace=False)           
    # internal points with wall BC
    xg_train = np.concatenate((xyu_inlet[:,0:1],xyu_wall[:,0:1],xyu_outlet_t[:,0:1],xyu_int[idx,0:1]),axis=0)
    yg_train = np.concatenate((xyu_inlet[:,1:2],xyu_wall[:,1:2],xyu_outlet_t[:,1:2],xyu_int[idx,1:2]),axis=0)
    tg_train = np.concatenate((xyu_inlet[:,2:3],xyu_wall[:,2:3],xyu_outlet_t[:,2:3],xyu_int[idx,2:3]),axis=0)
        
    # Training
    model = PhysicsInformedNN(x_train, y_train, t_train, p_train, u_train, v_train, \
                              x_iw, y_iw, t_iw, p_iw, u_iw, v_iw, nx_iw, ny_iw, \
                              x_oo, y_oo, t_oo, p_oo, u_oo, v_oo, nx_oo, ny_oo, \
                              xg_train, yg_train, tg_train)
 
    model.train(100000,True)  
       
    model.save_model(000000)


############################
 
#plt.figure(figsize=(6, 4), dpi=100)
#plt0, =plt.plot(xyu_wall[:,0:1],xyu_wall[:,1:2],'ok',linewidth=0,ms=3,label='MSE BC pts: 800',zorder=5)
#plt0, =plt.plot(xyu_outlet_t[:,0:1],xyu_outlet_t[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
#plt0, =plt.plot(xyu_inlet[:,0:1],xyu_inlet[:,1:2],'ok',linewidth=0,ms=3,zorder=5)
#plt0, =plt.plot(xyu_int[:,0:1],xyu_int[:,1:2],'+r',linewidth=0,ms=2,label='Gov Eq. Res. pts: 20000 ',zorder=4)
#plt0, =plt.plot(xyu_s[:,0:1],xyu_s[:,1:2],'og',linewidth=0,ms=4,label='MSE internal pts: 54 ',zorder=8)
#
##
####text-1
##plt.text(2.5, -0.3, "Wall: u=0", horizontalalignment='center', verticalalignment='center')
##plt.text(2.5, 3.3, "Outlet: p=0", horizontalalignment='center', verticalalignment='center')
##plt.text(-0.3, 1.5, "Inlet: u-specified", horizontalalignment='center', verticalalignment='center',rotation=90)
#
##plt.legend(fontsize=20)
#plt.xlabel('X',fontsize=20)
#plt.ylabel('Y',fontsize=20)
##plt.title('%s-u'%(flist[ii]),fontsiuze=16)
#plt.legend(loc='upper center', bbox_to_anchor=(1.45, 1), ncol=1, fancybox=False, shadow=False,fontsize=16)
#plt.xlim(-3,3)
#plt.ylim(-3,3)    
#plt.savefig('./plot/mesh9.png', format='png',bbox_inches='tight', dpi=200)
#plt.show()

