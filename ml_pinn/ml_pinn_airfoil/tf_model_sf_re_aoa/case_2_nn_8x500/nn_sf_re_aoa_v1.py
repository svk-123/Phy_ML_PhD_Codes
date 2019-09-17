#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:38:15 2019

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
    def __init__(self, x, y, r, a, para, u, v, p):
                  
        self.x = x
        self.y = y
        self.r = r
        self.para = para
        self.u = u
        self.v = v
        self.p = p
        self.a = a
        
        # Initialize parameters (1/2000)
        self.nu = tf.constant([0.0005], dtype=tf.float32)
        
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]],name='input0')
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]],name='input1')
        self.r_tf = tf.placeholder(tf.float32, shape=[None, self.r.shape[1]],name='input2')
        self.a_tf = tf.placeholder(tf.float32, shape=[None, self.a.shape[1]],name='input3')
        self.para_tf = tf.placeholder(tf.float32, shape=[None, self.para.shape[1]],name='input4')
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        self.p_tf = tf.placeholder(tf.float32, shape=[None, self.p.shape[1]])
         
        self.u_pred, self.v_pred, self.p_pred = self.net_NS(self.x_tf,\
                                                            self.y_tf, self.r_tf, self.a_tf, self.para_tf)
        
        
#        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
#                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
#                    tf.reduce_sum(tf.square(self.p_tf - self.p_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_c_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_v_pred))
                    
        self.loss_1 = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_mean(tf.square(self.p_tf - self.p_pred))
                    
        self.loss_2 = 0.0
                    
        self.loss = self.loss_1 
                    
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
        l1 = tf.layers.dense(X,  500, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 500, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 500, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 500, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 500, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 500, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 500, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 500, activation=tf.nn.tanh)
        Y  = tf.layers.dense(l1,3,activation=None,name='prediction')
        
        
        return Y
        
    def net_NS(self, x, y, r, a, para):

        uvp = self.neural_net(tf.concat([x,y,r,a,para], 1))
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
      
        
        return u, v, p
    
    def callback(self, loss, loss_1, loss_2):
        print('Loss: %.6e %.6e %.6e \n' % (loss,loss_1,loss_2))       
        self.fp.write('00, %.6e, %.6e, %.6e \n'% (loss,loss_1,loss_2)) 
        
    def get_batch(self,idx,bs,tb):
        
        return self.x[idx*bs:(idx+1)*bs],self.y[idx*bs:(idx+1)*bs],self.r[idx*bs:(idx+1)*bs],self.a[idx*bs:(idx+1)*bs],\
                self.para[idx*bs:(idx+1)*bs],self.u[idx*bs:(idx+1)*bs],self.v[idx*bs:(idx+1)*bs],self.p[idx*bs:(idx+1)*bs]
      
    def train(self, nIter, lbfgs=False): 
        
        self.fp=open('./tf_model/conv.dat','w')
    
    
        batch_size=1000
        total_batch= self.x.shape[0] / batch_size
        if(self.x.shape[0] % batch_size != 0):
            total_batch = total_batch +1
        print('total batch',total_batch)
        
        lr=0.001
        min_lr=1e-6
        #reduce lr iter(patience)
        rli=200
        #numbers to avg
        L=30
        #lr eps
        l_eps=1e-4
        
        #early stop wait
        estop=1000
        e_eps=1e-5
        
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
            
                batch_x, batch_y, batch_r, batch_a, batch_para, batch_u, batch_v, batch_p = self.get_batch(i,batch_size,total_batch)
                
                tf_dict = {self.x_tf: batch_x, self.y_tf: batch_y, self.r_tf: batch_r, self.a_tf: batch_a, self.para_tf: batch_para,
                   self.u_tf: batch_u, self.v_tf: batch_v, self.p_tf: batch_p, self.tf_lr:lr}
            
                _,loss_value=self.sess.run([self.train_op_Adam,self.loss], tf_dict)
                avg_loss += loss_value / total_batch
  
            
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
                          %(count, avg_loss,avg_lv_1, avg_lv_2,lr, elapsed))
            
            self.fp.write('%d, %.6e, %0.6e, %0.6e, %0.6e, %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2,lr, elapsed))    
            start_time = time.time()
            
            #save model
            if ((count % 200) ==0):
                model.save_model(count)
                
       
        #final_optimization using lbfgsb
        if (lbfgs==True):
                    
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.r_tf: self.r, self.a_tf: self.a,
                       self.u_tf: self.u, self.v_tf: self.v, self.p_tf: self.p}            
                
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss,self.loss_1,self.loss_2],
                                    loss_callback = self.callback)
 

        self.fp.close()
            
    
    def predict(self, x_star, y_star, r_star, aoa_star, para_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star,self.r_tf: r_star,\
                   self.aoa_tf: aoa_star, self.para_tf: para_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, v_star, p_star

    def save_model(self,count):
        self.saver.save(self.sess,'./tf_model/model_%d'%count)        
        
if __name__ == "__main__": 
      
           
    # Load Data
    #load data
    inp_x=[]
    inp_y=[]
    inp_reno=[]
    inp_aoa=[]
    inp_para=[]

    out_p=[]
    out_u=[]
    out_v=[]
    
    for ii in range(1,12):
        #x,y,Re,u,v
        with open('../../ml_para_sf/data_file/foil_nacan_lam_tr_%s.pkl'%ii, 'rb') as infile:
            result = pickle.load(infile)
        inp_x.extend(result[0])   
        inp_y.extend(result[1])
        inp_para.extend(result[2])
        inp_reno.extend(result[3])
        inp_aoa.extend(result[4])

        out_p.extend(result[5])
        out_u.extend(result[6])
        out_v.extend(result[7])
        

        
    inp_x=np.asarray(inp_x)
    inp_y=np.asarray(inp_y)
    inp_reno=np.asarray(inp_reno)
    inp_aoa=np.asarray(inp_aoa)
    inp_para=np.asarray(inp_para)

    out_p=np.asarray(out_p)
    out_u=np.asarray(out_u)
    out_v=np.asarray(out_v)
    
    inp_x = inp_x[:,None] # NT x 1
    inp_y = inp_y[:,None] # NT x 1
    inp_reno = inp_reno[:,None]/2000.
    inp_aoa = inp_aoa[:,None]/16.
    inp_para = inp_para[:,:]
    
    out_u = out_u[:,None] # NT x 1
    out_v = out_v[:,None] # NT x 1
    out_p = out_p[:,None] # NT x 1


    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    
    N_train=5000*50
    
    idx = np.random.choice(len(inp_x), N_train, replace=False)
    x_train = inp_x[idx,:]
    y_train = inp_y[idx,:]
    r_train = inp_reno[idx,:]
    aoa_train = inp_aoa[idx,:]
    para_train = inp_para[idx,:]
    
    u_train = out_u[idx,:]
    v_train = out_v[idx,:]
    p_train = out_p[idx,:]

    del result
    del inp_x
    del inp_y
    del inp_para
    del inp_aoa
    del inp_reno
    
    del out_u
    del out_v
    del out_p
        
    # Training
    model = PhysicsInformedNN(x_train, y_train, r_train, aoa_train, para_train, u_train, v_train, p_train)
 
    model.train(100000)  
       
    model.save_model(000000)

    print("--- %s seconds ---" % (time.time() - start_time))
    
  

             
    


