#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:49:13 2019

@author: vino
"""

"""
@author: Maziar Raissi
"""
'''
this is to make prediction using
p u v instead of original psi_p work
lamda removed
Re based training added
lr variable added
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
    def __init__(self, x, y, r, u, v, p,rst=False):
        
          
        self.x = x
        self.y = y
        self.r = r
        
        self.u = u
        self.v = v
        self.p = p
        
        # Initialize parameters
        self.nu = tf.constant([0.001], dtype=tf.float32)
        
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]],name='input0')
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]],name='input1')
        self.r_tf = tf.placeholder(tf.float32, shape=[None, self.r.shape[1]],name='input2')
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        self.p_tf = tf.placeholder(tf.float32, shape=[None, self.p.shape[1]])
         
        self.u_pred, self.v_pred, self.p_pred, self.f_c_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf, self.r_tf)
        
#        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
#                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
#                    tf.reduce_sum(tf.square(self.p_tf - self.p_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_c_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_v_pred))
                    
        self.loss_1 = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_mean(tf.square(self.p_tf - self.p_pred))
                    
        self.loss_2 = tf.reduce_mean(tf.square(self.f_c_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred))
                    
        self.loss = self.loss_1 + self.loss_2
            
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 100,
                                                                           'maxls': 100,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        
        
        self.train_op_Adam = tf.train.AdamOptimizer(self.tf_lr).minimize(self.loss)                    
        self.train_op_Adam_1 = tf.train.AdamOptimizer(self.tf_lr).minimize(self.loss_1)        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        self.saver = tf.train.Saver()
        
        if(rst == True):
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./tf_model/'))
        
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
        
    def net_NS(self, x, y, r):

        uvp = self.neural_net(tf.concat([x,y,r], 1))
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
      
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_c =  u_x + v_y
        f_u =  (u*u_x + v*u_y) + p_x - (self.nu/r)*(u_xx + u_yy) 
        f_v =  (u*v_x + v*v_y) + p_y - (self.nu/r)*(v_xx + v_yy)
        
        return u, v, p, f_c, f_u, f_v
    
    def callback(self, loss, loss_1, loss_2):
        print('Loss: %.6e %.6e %.6e \n' % (loss,loss_1,loss_2))       
        self.fp.write('00, %.6e, %.6e, %.6e \n'% (loss,loss_1,loss_2)) 
        
    def get_batch(self,idx,bs,tb):
        
        return self.x[idx*bs:(idx+1)*bs],self.y[idx*bs:(idx+1)*bs],self.r[idx*bs:(idx+1)*bs],\
                self.u[idx*bs:(idx+1)*bs],self.v[idx*bs:(idx+1)*bs],self.p[idx*bs:(idx+1)*bs]
      
    def train(self, nIter, lbfgs=False): 
        
        self.fp=open('./tf_model_1/conv.dat','w')
        
        batch_size=50
        total_batch= self.x.shape[0] / batch_size
        if(self.x.shape[0] % batch_size != 0):
            total_batch = total_batch +1
        print('total batch',total_batch)
        
        lr=1e-6
        min_lr=1e-7
        #reduce lr iter(patience)
        rli=200
        l_eps=1e-6
        #numbers to avg
        L=30
        #early stop wait
        estop=1000
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
            
            #batch training
            for i in range(total_batch):
            
                batch_x, batch_y, batch_r, batch_u, batch_v, batch_p = self.get_batch(i,batch_size,total_batch)
                
                tf_dict = {self.x_tf: batch_x, self.y_tf: batch_y, self.r_tf: batch_r,
                   self.u_tf: batch_u, self.v_tf: batch_v, self.p_tf: batch_p, self.tf_lr:lr}
                _,loss_value,lv_1,lv_2=self.sess.run([self.train_op_Adam,self.loss,self.loss_1,self.loss_2], tf_dict)
                avg_loss += loss_value / total_batch
                avg_lv_1 += lv_1 / total_batch
                avg_lv_2 += lv_2 / total_batch          
                
            my_hist.append(avg_loss)
            
            #reduce lr
            if(len(my_hist) > rli  and lr > min_lr):
                if ((sum(my_hist[-rli:-rli+L]) - sum(my_hist[-L-1:-1])) < (l_eps*L) ):
                    lr=lr*0.2
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
                          %(count, avg_loss,avg_lv_1, avg_lv_2,lr, elapsed))
            
            self.fp.write('%d, %.6e, %0.6e, %0.6e, %0.6e, %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2,lr, elapsed))    
            start_time = time.time()
            
            #save model
            if ((count % 1000) ==0):
                model.save_model(count)
                
       
        #final_optimization using lbfgsb
        if (lbfgs==True):
                    
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.r_tf: self.r,
                       self.u_tf: self.u, self.v_tf: self.v, self.p_tf: self.p}            
                
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss,self.loss_1,self.loss_2],
                                    loss_callback = self.callback)
 

        self.fp.close()
            
    
    def predict(self, x_star, y_star,r_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star,self.r_tf: r_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, v_star, p_star

    def save_model(self,count):
           
        self.saver.save(self.sess,'./tf_model_1/model_%d'%count)        
        
if __name__ == "__main__": 
      
           
    # Load Data
    #load data
    xtmp=[]
    ytmp=[]
    reytmp=[]
    utmp=[]
    vtmp=[]
    ptmp=[]
    
    #relist=[100,200,400,600,1000,2000,4000,6000,8000,9000]
    relist=[100,200,300,400,600,700,800,900]
    for ii in range(len(relist)):
        #x,y,Re,u,v
        with open('../../data_file_ldc_ust/cavity_Re%s.pkl'%relist[ii], 'rb') as infile:
            result = pickle.load(infile)
        xtmp.extend(result[0])
        ytmp.extend(result[1])
        reytmp.extend(result[2])
        utmp.extend(result[3])
        vtmp.extend(result[4])
        ptmp.extend(result[5])   
        
    xtmp=np.asarray(xtmp)
    ytmp=np.asarray(ytmp)
    utmp=np.asarray(utmp)
    vtmp=np.asarray(vtmp)
    ptmp=np.asarray(ptmp) 
    reytmp=np.asarray(reytmp)/1000.    
       
    x = xtmp[:,None] # NT x 1
    y = ytmp[:,None] # NT x 1
    
    u = utmp[:,None] # NT x 1
    v = vtmp[:,None] # NT x 1
    p = ptmp[:,None] # NT x 1
    r = reytmp[:,None]    
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    
    N_train=800
    
    idx = np.random.choice(len(xtmp), N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    r_train = r[idx,:]
    
    u_train = u[idx,:]
    v_train = v[idx,:]
    p_train = p[idx,:]

    
    # Training
    model = PhysicsInformedNN(x_train, y_train, r_train, u_train, v_train, p_train, True)
 
    model.train(100,True)  
       
    model.save_model(000000)

    print("--- %s seconds ---" % (time.time() - start_time))
    
  

             
    


