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
    def __init__(self, x, y, t, u, v, p, xb, yb, tb, ub, vb, xg, yg, tg, rst=False):
                  
        self.x = x
        self.y = y
        self.t = t
        
        self.u = u
        self.v = v
        self.p = p

        self.xb = xb
        self.yb = yb
        self.tb = tb
        
        self.ub = ub
        self.vb = vb
        
        self.xg = xg
        self.yg = yg
        self.tg = tg        
        
        # Initialize parameters (1/2000)
        self.nu = tf.constant([0.01], dtype=tf.float32)
        
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
        
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.yb_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        self.tb_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        
        self.ub_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        self.vb_tf = tf.placeholder(tf.float32, shape=[None, self.vb.shape[1]]) 

       
        self.xg_tf = tf.placeholder(tf.float32, shape=[None, self.xg.shape[1]])
        self.yg_tf = tf.placeholder(tf.float32, shape=[None, self.yg.shape[1]])
        self.tg_tf = tf.placeholder(tf.float32, shape=[None, self.yg.shape[1]])
        
        self.u_pred, self.v_pred, self.p_pred  = self.net_NS1(self.x_tf, self.y_tf, self.t_tf)
        
        self.ub_pred, self.vb_pred, _ = self.net_NS2(self.xb_tf, self.yb_tf, self.tb_tf)
        
        self.f_c_pred, self.f_u_pred, self.f_v_pred = self.net_NS3(self.xg_tf, self.yg_tf, self.tg_tf)

        
#        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
#                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
#                    tf.reduce_sum(tf.square(self.p_tf - self.p_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_c_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_v_pred))
                    
        self.loss_1 = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_mean(tf.square(self.p_tf - self.p_pred)) 


        self.loss_2 = tf.reduce_mean(tf.square(self.ub_tf - self.ub_pred)) + \
                    tf.reduce_mean(tf.square(self.vb_tf - self.vb_pred)) 

                    
        self.loss_3 = tf.reduce_mean(tf.square(self.f_c_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred))
                    
        self.loss = self.loss_1 + self.loss_2 + self.loss_3
                    
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
	self.lb_count=0

        if(rst == True):
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./tf_model_3/'))
      
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
        
    def net_NS1(self, x, y, t):
        
        with tf.variable_scope("NS1"):
            uvp = self.neural_net(tf.concat([x,y,t], 1))
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
      
        
        return u, v, p

    def net_NS2(self, x, y, t):
        
        with tf.variable_scope("NS1",reuse=True):
            uvp = self.neural_net(tf.concat([x,y,t], 1))
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
      
        
        return u, v, p    
    
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
    
    def callback(self, loss, loss_1, loss_2, loss_3):
        print('Loss: %.6e %.6e %.6e %.6e \n' % (loss,loss_1,loss_2, loss_3))       
        self.fp.write('00, %.6e, %.6e, %.6e %.6e \n'% (loss,loss_1,loss_2, loss_3)) 
        self.lb_count=self.lb_count+1
	if(self.lb_count % 500 ==0):
      	    self.save_model(self.lb_count)

    def train(self, nIter, lbfgs=False): 
        
        self.fp=open('./tf_model/conv.dat','w')
    
    

        total_batch= 1.0

        
        lr=0.000001
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
            avg_lv_3 = 0.
            
            #batch training
            for i in range(1):
                            
                
                tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t, self.u_tf: self.u, self.v_tf: self.v, self.p_tf: self.p, \
                           self.xb_tf: self.xb, self.yb_tf: self.yb, self.tb_tf: self.tb, self.ub_tf: self.ub, self.vb_tf: self.vb, \
                           self.tf_lr:lr, self.xg_tf: self.xg, self.yg_tf: self.yg,  self.tg_tf: self.tg}
            
                _,loss_value,lv_1,lv_2,lv_3=self.sess.run([self.train_op_Adam,self.loss,self.loss_1,self.loss_2,self.loss_3], tf_dict)
                avg_loss += loss_value / total_batch
                avg_lv_1 += lv_1 / total_batch
                avg_lv_2 += lv_2 / total_batch   
                avg_lv_3 += lv_3 / total_batch 
                
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
            print('It: %d, Loss: %.6e, Loss-1:%0.6e, Loss-2:%0.6e, Loss-2:%0.6e, lr:%0.6f, Time: %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2, avg_lv_3, lr, elapsed))
            
            self.fp.write('%d, %.6e, %0.6e, %0.6e, %0.6e, %0.6e, %.2f \n' \
                          %(count, avg_loss,avg_lv_1, avg_lv_2, avg_lv_3, lr, elapsed))    
            start_time = time.time()
            
            #save model
            if ((count % 2000) ==0):
                self.save_model(count)
                
       
        #final_optimization using lbfgsb
        if (lbfgs==True):
                    
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t, self.u_tf: self.u, self.v_tf: self.v, self.p_tf: self.p, \
                           self.xb_tf: self.xb, self.yb_tf: self.yb, self.tb_tf: self.tb, self.ub_tf: self.ub, self.vb_tf: self.vb, \
                           self.tf_lr:lr, self.xg_tf: self.xg, self.yg_tf: self.yg,  self.tg_tf: self.tg}       
                
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
           
    # Load Data
    #load data
    inp_x=[]
    inp_y=[]
    inp_reno=[]
    inp_aoa=[]
    inp_para=[]
    inp_t=[]
    
    out_p=[]
    out_u=[]
    out_v=[]
    
    for ii in range(1):
        #x,y,Re,u,v
        with open('../data_file/cy_un_lam_around_5555_1.pkl', 'rb') as infile:
            result = pickle.load(infile)
        inp_x.extend(result[0])   
        inp_y.extend(result[1])
        inp_t.extend(result[3])
        
        out_p.extend(result[4])
        out_u.extend(result[5])
        out_v.extend(result[6])
        
        
    inp_x=np.asarray(inp_x)
    inp_y=np.asarray(inp_y)
    inp_t=np.asarray(inp_t)
    
    out_p=np.asarray(out_p)
    out_u=np.asarray(out_u)
    out_v=np.asarray(out_v)
    
    inp_x = inp_x[:,None].copy() # NT x 1
    inp_y = inp_y[:,None].copy() # NT x 1
    inp_t = inp_t[:,None].copy() # NT x 1    
    
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
    pinp_t=[]
    
    pout_p=[]
    pout_u=[]
    pout_v=[]
    for ii in range(1):
        #x,y,re,t,p,u,v
        with open('../data_file/cy_un_lam_around_5555_1.pkl', 'rb') as infile:
            result = pickle.load(infile)
        pinp_x.extend(result[0])   
        pinp_y.extend(result[1])
        pinp_t.extend(result[3])
        
        pout_p.extend(result[4])
        pout_u.extend(result[5])
        pout_v.extend(result[6])
        
        
    pinp_x=np.asarray(pinp_x)
    pinp_y=np.asarray(pinp_y)
    pinp_t=np.asarray(pinp_t)

    pout_p=np.asarray(pout_p)
    pout_u=np.asarray(pout_u)
    pout_v=np.asarray(pout_v)
    
    pinp_x = pinp_x[:,None].copy() # NT x 1
    pinp_y = pinp_y[:,None].copy() # NT x 1
    pinp_t = pinp_t[:,None].copy() # NT x 1
    
    pout_u = pout_u[:,None].copy() # NT x 1
    pout_v = pout_v[:,None].copy() # NT x 1
    pout_p = pout_p[:,None].copy() # NT x 1

    
    ######################################################################
    ######################## MSE Data ###############################
    ######################################################################
    # Training Data    
    
    #import wall bc
    xyu=np.loadtxt('../data_file/cy_wall_bc_20_t.dat')
    
    xyu_io=np.loadtxt('../data_file/cy_inout_20_t.dat')
    
    N_train=5000
    
    idx = np.random.choice(len(inp_x), N_train, replace=False)
    
    #internal points
    x_train = inp_x[idx,:]
    y_train = inp_y[idx,:]
    t_train = inp_t[idx,:]
    
    u_train = out_u[idx,:]
    v_train = out_v[idx,:]
    p_train = out_p[idx,:]
    
    
    # only wall BC
    tb_train = np.concatenate((xyu[:,0:1], xyu_io[:,0:1]),axis=0)    
    xb_train = np.concatenate((xyu[:,1:2], xyu_io[:,1:2]),axis=0)
    yb_train = np.concatenate((xyu[:,2:3], xyu_io[:,2:3]),axis=0)
    ub_train = np.concatenate((xyu[:,3:4], xyu_io[:,3:4]),axis=0)
    vb_train = np.concatenate((xyu[:,4:5], xyu_io[:,4:5]),axis=0)
    
    ######################################################################
    ######################## Gov Data ###############################
    ######################################################################
    
    N_train=150000
    
    idx = np.random.choice(len(pinp_x), N_train, replace=False)
    
    # internal points with wall BC
    xg_train = np.concatenate((x_train[:,:],xb_train[:,:],pinp_x[idx,:]),axis=0)
    yg_train = np.concatenate((y_train[:,:],yb_train[:,:],pinp_y[idx,:]),axis=0)
    tg_train = np.concatenate((t_train[:,:],tb_train[:,:],pinp_t[idx,:]),axis=0)     
    
    # Training
    model = PhysicsInformedNN(x_train, y_train, t_train,  u_train, v_train, p_train, xb_train, yb_train, tb_train, ub_train, vb_train, xg_train, yg_train, tg_train, True)
 
    model.train(1, True)  
       
    model.save_model(000000)
    
#    # Prediction
#    u_pred, v_pred, p_pred = model.predict(pinp_x, pinp_y)
#    
#    print("--- %s seconds ---" % (time.time() - start_time))
#    
#  
#    plt.figure(figsize=(6,5),dpi=100)
#    plt.plot(inp_x[idx,:],inp_y[idx,:],'o')
#    plt.plot(xyu[:,0],xyu[:,1],'k')    
#    plt.savefig('cy_points.png',format='png',dpi=300)
#    plt.show()
#             
#    plt.figure(figsize=(6,5),dpi=100)
#    plt.contourf(pinp_x[:,0],pinp_y[:,0],u_pred[:,0])
#    #plt.plot(xyu[:,0],xyu[:,1],'k')    
#    plt.savefig('contour.png',format='png',dpi=300)
#    plt.show()    


#plt.figure(figsize=(6, 5), dpi=100)
##plt0, =plt.plot(x_train,y_train,'og',linewidth=0,ms=5,label='MSE pts-200 (Sampling)',zorder=5)
##plt0, =plt.plot(xg_train,yg_train,'+r',linewidth=0,ms=3,label='Gov Eq. pts-8000 (Residual)',zorder=0)
#plt0, =plt.plot(xb_train,yb_train,'ok',linewidth=0,ms=4,label='BC pts-80',zorder=1)
#
#plt0, =plt.plot([0,0],[0.5,1],'r',linewidth=2)
#plt0, =plt.plot([1,1],[0,1],'r',linewidth=2)
#plt0, =plt.plot([2,2],[0,1],'r',linewidth=2)
#
##plt.legend(fontsize=20)
#plt.xlabel('X',fontsize=20)
#plt.ylabel('Y',fontsize=20)
##plt.title('%s-u'%(flist[ii]),fontsiuze=16)
##plt.legend(loc='upper center', bbox_to_anchor=(1.45, 1), ncol=1, fancybox=False, shadow=False,fontsize=16)
##plt.xlim(-0.1,1.2)
##plt.ylim(-0.01,1.4)    
#plt.savefig('./plot/mesh_4.png', format='png',bbox_inches='tight', dpi=100)
#plt.show()
