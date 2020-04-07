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
    def __init__(self, x, y,rst=False):
        
          
        self.x = x
        self.y = y

        
        # Initialize parameters
        self.nu = tf.constant([0.0001], dtype=tf.float32)
        
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1],self.x.shape[2],self.x.shape[3]],name='input')
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]],name='output')
         
        self.y_pred = self.net_NS(self.x_tf)
        
                    
        self.loss_1 = tf.reduce_mean(tf.square(self.y_tf - self.y_pred))

                    
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
        
        if(rst == True):
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./tf_model/'))
        
    def neural_net(self, X):

        #create model
        conv1 = tf.layers.conv2d(X, 32, 8, activation=tf.nn.tanh)
        pool1 = tf.layers.max_pooling2d(conv1, 3, 3)
        conv2 = tf.layers.conv2d(pool1, 64, 6, activation=tf.nn.tanh)
        pool2= tf.layers.max_pooling2d(conv2, 3, 3)        
        conv3 = tf.layers.conv2d(pool2, 64, 4, activation=tf.nn.tanh)
        pool3= tf.layers.max_pooling2d(conv3, 2, 2)          
        conv4 = tf.layers.conv2d(pool3, 128, 2, activation=tf.nn.tanh)
        pool4= tf.layers.max_pooling2d(conv4, 2, 2)         
             
        flat1 = tf.contrib.layers.flatten(pool4)
        fc1= tf.layers.dense(flat1, 100, activation=tf.nn.tanh)
        fc2= tf.layers.dense(fc1, 100, activation=tf.nn.tanh)        
        fc3= tf.layers.dense(fc2, 100, activation=tf.nn.tanh)        
        fc4= tf.layers.dense(fc3, 8, activation=tf.nn.tanh, name='para')        
        fc5= tf.layers.dense(fc4, 100, activation=tf.nn.tanh)        
        fc6= tf.layers.dense(fc5, 100, activation=tf.nn.tanh)
        fc7= tf.layers.dense(fc6, 100, activation=tf.nn.tanh)
        Y = tf.layers.dense(fc7, 70,name='prediction')
                
        return Y
        
    def net_NS(self, x):

        uvp = self.neural_net(x)
                
        return uvp
    
    def callback(self, loss):
        print('Loss: %.6e 00 00 \n' % (loss))       
        self.fp.write('00, %.6e, 00, 00 \n'% (loss)) 
        
    def get_batch(self,idx,bs,tb):
        return self.x[idx*bs:(idx+1)*bs],self.y[idx*bs:(idx+1)*bs]
    
    def train(self, nIter, lbfgs=False): 
        
        self.fp=open('./tf_model/conv.dat','w')
        
        batch_size=64
        total_batch= self.x.shape[0] / batch_size
        if(self.x.shape[0] % batch_size != 0):
            total_batch = total_batch +1
        print('total batch',total_batch)
        
        lr=0.0001
        min_lr=1e-7
        #reduce lr iter(patience)
        rli=100
        l_eps=1e-6
        #numbers to avg
        L=30
        #early stop wait
        estop=500
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
            
                batch_x, batch_y = self.get_batch(i,batch_size,total_batch)
                
                tf_dict = {self.x_tf: batch_x, self.y_tf: batch_y, self.tf_lr:lr}
            
                _,loss_value=self.sess.run([self.train_op_Adam,self.loss], tf_dict)
                avg_loss += loss_value / total_batch
        
                
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
            print('It: %d, Loss: %.6e, Loss-1:%0.6f, Loss-2:%0.6f, lr:%0.6f, Time: %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2,lr, elapsed))
            
            self.fp.write('%d, %.6e, %0.6e, %0.6e, %0.6e, %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2,lr, elapsed))    
            start_time = time.time()
            
            #save model
            if ((count % 100) ==0):
                model.save_model(count,avg_loss)
        #last inter
        model.save_model(count,avg_loss)
       
        #final_optimization using lbfgsb
        if (lbfgs==True):
                    
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y }            
                
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)
            
            model.save_model(0000,avg_loss)
 

        self.fp.close()
            
    
    def predict(self, x_star, y_star,r_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star,self.r_tf: r_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, v_star, p_star

    def save_model(self,count,avg_loss):
           
        self.saver.save(self.sess,'./tf_model/model_%d_%0.6f'%(count,avg_loss))        
        
if __name__ == "__main__": 
      
           
    #load Data
    #load data
    xtmp=[]
    ytmp=[]
    reytmp=[]
    utmp=[]
    vtmp=[]
    ptmp=[]
    
    for ii in range(1):
        #x,y,Re,u,v
        with open('../data_file/foil_param_216_no_aug_tr.pkl', 'rb') as infile:
            result = pickle.load(infile)
        my_inp=result[0]
        my_out=result[1]
        xx=result[2]


    my_inp=np.asarray(my_inp)
    my_out=np.asarray(my_out)
    my_inp=my_inp[:,:,:,None]    
    my_out=my_out    

    # Training
    model = PhysicsInformedNN(my_inp,my_out,False)
 
    model.train(2000,True)  
       
    

    print("--- %s seconds ---" % (time.time() - start_time))
    
  

             
    


