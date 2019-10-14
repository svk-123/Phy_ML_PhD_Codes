#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:35:28 2019

@author: vino
"""

"""
@author: Maziar Raissi
"""
'''
checked for single network
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
    def __init__(self, x, y, u, v, xg, yg, layers):
                  
        self.x = x
        self.y = y

        self.u = u
        self.v = v
        #self.p = p
        
        self.xg = xg
        self.yg = yg
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers) 
        
        # Initialize parameters (1/2000)
        self.nu = tf.constant([0.025], dtype=tf.float32)
        
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]],name='input0')
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]],name='input1')
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        self.p_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        
        self.xg_tf = tf.placeholder(tf.float32, shape=[None, self.xg.shape[1]])
        self.yg_tf = tf.placeholder(tf.float32, shape=[None, self.yg.shape[1]])
        
        self.u_pred, self.v_pred, self.p_pred,self.f_c_pred, self.f_u_pred, self.f_v_pred   = self.net_NS1(self.x_tf, self.y_tf)


        
#        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
#                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
#                    tf.reduce_sum(tf.square(self.p_tf - self.p_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_c_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_v_pred))
                    
        self.loss_1 = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) 
                    
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
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
    
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    
    
    def net_NS1(self, x, y):

        uvp = self.neural_net(tf.concat([x,y], 1),self.weights, self.biases)
        
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
        f_u =  (u*u_x + v*u_y) + p_x - (self.nu)*(u_xx + u_yy) 
        f_v =  (u*v_x + v*v_y) + p_y - (self.nu)*(v_xx + v_yy)
        
        return  u,v,p,f_c, f_u, f_v
    
    def callback(self, loss, loss_1, loss_2):
        print('Loss: %.6e %.6e %.6e \n' % (loss,loss_1,loss_2))       
        self.fp.write('00, %.6e, %.6e, %.6e \n'% (loss,loss_1,loss_2)) 
        
    def get_batch(self,idx,bs,tb):
        
        return self.x[idx*bs:(idx+1)*bs],self.y[idx*bs:(idx+1)*bs],self.u[idx*bs:(idx+1)*bs],\
               self.v[idx*bs:(idx+1)*bs]
      
    def train(self, nIter, lbfgs=False): 
        
        self.fp=open('./tf_model/conv.dat','w')
    
    
        batch_size=1000
        total_batch= self.x.shape[0] / batch_size
        if(self.x.shape[0] % batch_size != 0):
            total_batch = total_batch +1
        print('total batch',total_batch)
        
        lr=0.0001
        min_lr=1e-7
        #reduce lr iter(patience)
        rli=500
        #numbers to avg
        L=30
        #lr eps
        l_eps=1e-7
        
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
            
                batch_x, batch_y, batch_u, batch_v = self.get_batch(i,batch_size,total_batch)
                
                tf_dict = {self.x_tf: batch_x, self.y_tf: batch_y, self.u_tf: batch_u, self.v_tf: batch_v, self.tf_lr:lr}
                          
            
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
                          %(count, avg_loss,avg_lv_1, avg_lv_2,lr, elapsed))
            
            self.fp.write('%d, %.6e, %0.6e, %0.6e, %0.6e, %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2,lr, elapsed))    
            start_time = time.time()
            
            #save model
            if ((count % 5000) ==0):
                model.save_model(count)
                
       
        #final_optimization using lbfgsb
        if (lbfgs==True):
                    
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.u_tf: self.u, self.v_tf: self.v}            
                
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
    layers = [2, 100, 100, 100, 100, 100, 100, 100, 100, 3]  
           
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
    
    for ii in range(1):
        #x,y,Re,u,v
        with open('./data_file/cy_40_wake.pkl', 'rb') as infile:
            result = pickle.load(infile)
        inp_x.extend(result[0])   
        inp_y.extend(result[1])

        out_p.extend(result[2])
        out_u.extend(result[3])
        out_v.extend(result[4])
        
        
    inp_x=np.asarray(inp_x)
    inp_y=np.asarray(inp_y)
    out_p=np.asarray(out_p)
    out_u=np.asarray(out_u)
    out_v=np.asarray(out_v)
    
    inp_x = inp_x[:,None].copy() # NT x 1
    inp_y = inp_y[:,None].copy() # NT x 1
    out_u = out_u[:,None].copy() # NT x 1
    out_v = out_v[:,None].copy() # NT x 1
    out_p = out_p[:,None].copy() # NT x 1
    
    # Load Data
    #load data
    pinp_x=[]
    pinp_y=[]
    pinp_reno=[]
    pinp_aoa=[]
    pinp_para=[]

    pout_p=[]
    pout_u=[]
    pout_v=[]
    for ii in range(1):
        #x,y,Re,u,v
        with open('./data_file/cy_40_around.pkl', 'rb') as infile:
            result = pickle.load(infile)
        pinp_x.extend(result[0])   
        pinp_y.extend(result[1])

        pout_p.extend(result[2])
        pout_u.extend(result[3])
        pout_v.extend(result[4])
        
        
    pinp_x=np.asarray(pinp_x)
    pinp_y=np.asarray(pinp_y)

    pout_p=np.asarray(pout_p)
    pout_u=np.asarray(pout_u)
    pout_v=np.asarray(pout_v)
    
    pinp_x = pinp_x[:,None].copy() # NT x 1
    pinp_y = pinp_y[:,None].copy() # NT x 1
    pout_u = pout_u[:,None].copy() # NT x 1
    pout_v = pout_v[:,None].copy() # NT x 1
    pout_p = pout_p[:,None].copy() # NT x 1

    ######################################################################
    ######################## Gov Data ###############################
    ######################################################################
    #import wall bc
    xyu=np.loadtxt('./data_file/cy_wall_bc_50.dat')
    
    N_train=500
    
    idx = np.random.choice(len(pinp_x), N_train, replace=False)
    
    #with wall BC
    xg_train = np.concatenate((pinp_x[idx,:],xyu[:,0:1]),axis=0)
    yg_train = np.concatenate((pinp_y[idx,:],xyu[:,1:2]),axis=0)


    
    ######################################################################
    ######################## MSE Data ###############################
    ######################################################################
    # Training Data    
    
    #import wall bc
    xyu=np.loadtxt('./data_file/cy_wall_bc_50.dat')
    
    N_train=200
    
    idx = np.random.choice(len(inp_x), N_train, replace=False)
    
#    #without wall BC
#    x_train = inp_x[idx,:]
#    y_train = inp_y[idx,:]
#    u_train = out_u[idx,:]
#    v_train = out_v[idx,:]

    #with wall BC
    x_train = np.concatenate((inp_x[idx,:],xyu[:,0:1]),axis=0)
    y_train = np.concatenate((inp_y[idx,:],xyu[:,1:2]),axis=0)
    u_train = np.concatenate((out_u[idx,:],xyu[:,2:3]),axis=0)
    v_train = np.concatenate((out_v[idx,:],xyu[:,2:3]),axis=0)
        
    # Training
    model = PhysicsInformedNN(x_train, y_train, u_train, v_train, xg_train, yg_train , layers)
 
    model.train(1000,True)  
       
    model.save_model(000000)
    
    # Prediction
    u_pred, v_pred, p_pred = model.predict(pinp_x, pinp_y)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
  
    plt.figure(figsize=(6,5),dpi=100)
    plt.plot(inp_x[idx,:],inp_y[idx,:],'o')
    plt.plot(xyu[:,0],xyu[:,1],'k')    
    plt.savefig('cy_points.png',format='png',dpi=300)
    plt.show()
             
    plt.figure(figsize=(6,5),dpi=100)
    plt.tricontourf(pinp_x[:,0],pinp_y[:,0],u_pred[:,0])
    #plt.plot(xyu[:,0],xyu[:,1],'k')    
    plt.savefig('contour.png',format='png',dpi=300)
    plt.show()    


