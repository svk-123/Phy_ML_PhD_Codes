#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:35:28 2019

@author: vino
"""

"""
@author: originnaly written by Maziar Raissi
Further modified by Vinothkumar S.
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
    def __init__(self, x_iw, y_iw, u_iw,\
                              xg, yg):
                  
        self.x_iw = x_iw
        self.y_iw = y_iw
        self.u_iw = u_iw
                  
        self.xg = xg
        self.yg = yg
                
        # Initialize parameters (1/100)
        self.nu = tf.constant([0.01], dtype=tf.float32)
        
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_iw_tf = tf.placeholder(tf.float32, shape=[None, self.x_iw.shape[1]], name='input1a')
        self.y_iw_tf = tf.placeholder(tf.float32, shape=[None, self.y_iw.shape[1]], name='input1b')
        self.u_iw_tf = tf.placeholder(tf.float32, shape=[None, self.u_iw.shape[1]])
 
          
        self.xg_tf = tf.placeholder(tf.float32, shape=[None, self.xg.shape[1]])
        self.yg_tf = tf.placeholder(tf.float32, shape=[None, self.yg.shape[1]])
        
        #self.u_pred, self.v_pred, self.p_pred  = self.net_NS0(self.x_tf, self.y_tf)
        
        #MSE wall
        self.u_iw_pred = self.net_NS1(self.x_iw_tf, self.y_iw_tf)
        
        #gov Eqn
        self.f_u_pred = self.net_NS3(self.xg_tf, self.yg_tf)
       
#        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
#                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
#                    tf.reduce_sum(tf.square(self.p_tf - self.p_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_c_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_v_pred))
                    
#        self.loss_00 = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
#                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
#                    tf.reduce_mean(tf.square(self.p_tf - self.p_pred)) 

        self.loss_1 = tf.reduce_mean(tf.square(self.u_iw_tf - self.u_iw_pred))
       
        self.loss_2 = tf.reduce_mean(tf.square(self.f_u_pred)) 

                    
        self.loss = self.loss_1 + self.loss_2
                    
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
        Y  = tf.layers.dense(l1,1,activation=None,name='prediction')
                
        return Y
        
    def net_NS1(self, x, y):
        
        with tf.variable_scope("NS1"):
            uvp = self.neural_net(tf.concat([x,y], 1))
        
        u = uvp[:,0:1]
               
        return u


   
    def net_NS3(self, x, y):
        
        with tf.variable_scope("NS1",reuse=True):
            uvp = self.neural_net(tf.concat([x,y], 1))
        
        u = uvp[:,0:1]
      
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        
        f_u =  u_xx + u_yy

        return  f_u
    
    def callback(self, loss, loss_1, loss_2):
        print('Loss: %.6e %.6e %.6e \n' % (loss,loss_1,loss_2))       
        self.fp.write('00, %.6e, %.6e, %.6e  \n'% (loss,loss_1,loss_2)) 
        
      
    def train(self, nIter, lbfgs=False): 
        
        self.fp=open('./tf_model/conv.dat','w')
    
    

        total_batch= 1.0

        
        lr=0.001
        min_lr=1e-7
        #reduce lr iter(patience)
        rli=2000
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
                                            
                tf_dict = {self.x_iw_tf: self.x_iw, self.y_iw_tf: self.y_iw, self.u_iw_tf: self.u_iw, \
                           self.tf_lr:lr, self.xg_tf: self.xg, self.yg_tf: self.yg}
            
                _,loss_value,lv_1,lv_2=self.sess.run([self.train_op_Adam,self.loss,self.loss_1,self.loss_2], tf_dict)
                avg_loss += loss_value / total_batch
                avg_lv_1 += lv_1 / total_batch
                avg_lv_2 += lv_2 / total_batch   
                
            my_hist.append(avg_loss)
            
            #reduce lr
            if(len(my_hist) > rli  and lr > min_lr):
                if ((sum(my_hist[-rli:-rli+L]) - sum(my_hist[-L-1:-1])) < (l_eps*L) ):
                    lr=lr*0.5
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
            print('It: %d, Loss: %.6e, Loss-1:%0.6e, Loss-2:%0.6e, lr:%0.6f, Time: %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2,  lr, elapsed))
            
            self.fp.write('%d, %.6e, %0.6e, %0.6e, %0.6e, %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2,  lr, elapsed))    
            start_time = time.time()
            
            #save model
            if ((count % 5000) ==0):
                model.save_model(count)
                
       
        #final_optimization using lbfgsb
        if (lbfgs==True):
                    
            tf_dict = {self.x_iw_tf: self.x_iw, self.y_iw_tf: self.y_iw, self.u_iw_tf: self.u_iw, \
                           self.tf_lr:lr, self.xg_tf: self.xg, self.yg_tf: self.yg} 
                
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss,self.loss_1,self.loss_2],
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
    
    path='./data_file/'  
    #import wall bc
    #x,y,p,u,v
    xyu_inlet=np.loadtxt(path + 'laplace_boundary.dat')
    
    x_inlet = xyu_inlet[:,0:1]
    y_inlet = xyu_inlet[:,1:2]
    u_inlet = xyu_inlet[:,2:3] 
     
    ######################################################################
    ######################## Gov Data ####################################
    ######################################################################    
    
    xyu_int=np.loadtxt(path + 'laplace_internal_combined.dat',skiprows=1)    
                
    # internal points with wall BC
    xg_train = xyu_inlet[:,0:1]
    yg_train = xyu_inlet[:,1:2]
        
    # Training
    model = PhysicsInformedNN(x_inlet, y_inlet, u_inlet,\
                              xg_train, yg_train)
 
    model.train(10,True)  
       
    model.save_model(000000)
    
  
#plt.figure(figsize=(6, 6), dpi=100)
#plt0, =plt.plot(xyu_inlet[:,0:1],xyu_inlet[:,1:2],'ok',linewidth=0,ms=3,label='MSE BC pts: 800',zorder=5)
#plt0, =plt.plot(xyu_int[:,0:1],xyu_int[:,1:2],'+r',linewidth=0,ms=2,label='Gov Eq. Res. pts: 20000 ',zorder=4)
##plt.legend(fontsize=20)
#plt.xlabel('X',fontsize=20)
#plt.ylabel('Y',fontsize=20)
##plt.title('%s-u'%(flist[ii]),fontsiuze=16)
#plt.legend(loc='upper center', bbox_to_anchor=(1.45, 1), ncol=1, fancybox=False, shadow=False,fontsize=16)
##plt.xlim(-0.5,2)
##plt.ylim(-0.5,1)    
#plt.savefig('./plot/mesh2.png', format='png',bbox_inches='tight', dpi=100)
#plt.show()

