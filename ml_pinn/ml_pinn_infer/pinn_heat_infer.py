"""
@author: Maziar Raissi
"""
'''
this is to make prediction using
p u v instead of original psi_p work
lamda fixed
'''


import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import pickle

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self,x,tb,t):

        self.x = x
        self.tb = tb
        
        self.t = t
                                   
   
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1],name='ipt0')
        self.tb_tf = tf.placeholder(tf.float32, shape=[None, 1],name='ipt1')
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.A = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.Ep = tf.constant([0.0005], dtype=tf.float32)

        
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        self.t_pred, self.f_c_pred = self.net_NS(self.x_tf, self.tb_tf)
        
        self.loss = tf.reduce_sum(tf.square(self.t_tf - self.t_pred)) + \
                    tf.reduce_sum(tf.square(self.f_c_pred)) 

                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer(self.tf_lr)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        
        self.sess.run(init)

    
    def neural_net(self, X):

        #create model

        l1 = tf.layers.dense(X,  20, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 20, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 20, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 20, activation=tf.nn.tanh)
        Y  = tf.layers.dense(l1,2,activation=None,name='prediction')
        
        
        return Y
        
    def net_NS(self, x, tb):

        
        uvp = self.neural_net(tf.concat([x,tb], 1))
        
        u=uvp[:,0:1]
        A=uvp[:,1:2]
        
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        f_c =  u_xx - A * self.Ep * (tb**4 - u**4)

        
        return u, f_c
    
    def callback(self, loss):
        print('Loss: %.3e' % (loss))
        
    def get_batch(self,idx,bs,tb):
        
        return self.x[idx*bs:(idx+1)*bs],self.tb[idx*bs:(idx+1)*bs],self.t[idx*bs:(idx+1)*bs]

      
    def train(self, nIter, lbfgs=False): 
        
        fp=open('conv.dat','w')
        
        total_batch=3
        batch_size=50
        lr=0.001
        min_lr=1e-6
        #reduce lr iter(patience)
        rli=200
        #numbers to avg
        L=10
        #early stop wait
        estop=1000
        e_eps=1e-5
        
        start_time = time.time()
        
        my_hist=[]
        
        #epochs traings
        count=0
        while(count < nIter):
            count=count+1
            avg_loss = 0.
            
            #batch training
            for i in range(total_batch):
            
                batch_x, batch_tb, batch_t = self.get_batch(i,batch_size,total_batch)
                
                tf_dict = {self.x_tf: batch_x, self.tb_tf: batch_tb,
                   self.t_tf: batch_t,  self.tf_lr:lr}
            
                _,loss_value=self.sess.run([self.train_op_Adam,self.loss], tf_dict)
                avg_loss += loss_value / 1
            
            my_hist.append(avg_loss)
            
            #reduce lr
            if(len(my_hist) > rli  and lr > min_lr):
                if (sum(my_hist[-L-1:-1]) > sum(my_hist[-rli:-rli+L])):
                    lr=lr*0.5
                    print('Reduce Learning rate',lr,len(my_hist[-L-1:-1]),len(my_hist[-rli:-rli+L]))
                    my_hist=[]
                    
                    fp.write('Reduce Learning rate: %f \n' %lr)
                        
            #early stop        
            if(len(my_hist) > estop  and lr <= min_lr):
                if ( (sum(my_hist[-estop:-estop+L]) - sum(my_hist[-L-1:-1])) < (e_eps*L) ):
                    print ('Early STOP STOP STOP')
                    fp.write('Early STOP STOP STOP')
                    nIter=count
                    
            #print
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, Time: %.2f' %(count, avg_loss, elapsed))
            fp.write('It: %d, Loss: %.3e, Time: %.2f \n' %(count, avg_loss, elapsed))    
            start_time = time.time()
            
        
        #final_optimization using lbfgsb
        if (lbfgs==True):


                    
            tf_dict = {self.x_tf: self.x, self.tb_tf: self.tb,
                       self.t_tf: self.t}            
                
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)
 

        fp.close()
    
    def predict(self, x_star, y_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, v_star, p_star

    def save_model(self):
        
        saver = tf.train.Saver()
        saver.save(self.sess,'./tf_model/model') 
        
        
if __name__ == "__main__": 
      
            
    # Load Data
    #load data

    xtmp=[]
    ttmp=[]
    tbtmp=[]
    titmp=[]
    eptmp=[]
    
    for ii in range(3):
        #x,y,Re,u,v
        tmp=np.loadtxt('./1d_train1/T%d'%ii,delimiter=',')

        xtmp.extend(tmp[:,0])
        titmp.extend(tmp[:,1])
        tbtmp.extend(tmp[:,2])
        ttmp.extend(tmp[:,3])
        eptmp.extend(tmp[:,4])
    

    xtmp=np.asarray(xtmp)
    titmp=np.asarray(titmp)
    tbtmp=np.asarray(tbtmp)
    ttmp=np.asarray(ttmp)
    eptmp=np.asarray(eptmp)
       
    x  = xtmp[:,None] # NT x 1
    ti = titmp[:,None] # NT x 1
    
    tb = tbtmp[:,None] # NT x 1
    t  = ttmp[:,None] # NT x 1
    ep = eptmp[:,None] # NT x 1
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    N_train=150
    idx = np.random.choice(len(xtmp), N_train, replace=False)
    x_train = x[idx,:] 
    ti_train = ti[idx,:] 
    tb_train = tb[idx,:]
    t_train = t[idx,:] 


    # Training
    model = PhysicsInformedNN(x_train, ti_train, t_train)
    
    model.train(1000,True)
    
    model.save_model()
  

    
    
  

             
    


