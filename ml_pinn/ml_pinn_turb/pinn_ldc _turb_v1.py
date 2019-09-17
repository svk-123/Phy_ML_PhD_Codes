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
    def __init__(self,x,y,u,v):

        self.x = x
        self.y = y
        
        self.u = u
        self.v = v
                              
   
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1],name='input0')
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1],name='input1')
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.nu = tf.constant([0.00005], dtype=tf.float32)
                
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        self.u_pred, self.v_pred, self.p_pred, self.f_c_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf)
        
        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_sum(tf.square(self.f_c_pred)) + \
                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
                    tf.reduce_sum(tf.square(self.f_v_pred))
                    
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

        self.saver = tf.train.Saver()
    
    def neural_net(self, X):

        #create model
        l1 = tf.layers.dense(X,  20, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 20, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 20, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 20, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 20, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 20, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 20, activation=tf.nn.tanh)
        Y  = tf.layers.dense(l1,4,activation=None,name='prediction')
        
        
        return Y
        
    def net_NS(self, x, y):
        
        uvp = self.neural_net(tf.concat([x,y], 1))
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
        nut = uvp[:,3:4]

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
        f_u =  (u*u_x + v*u_y) + p_x - (self.nu + nut)*(u_xx + u_yy) 
        f_v =  (u*v_x + v*v_y) + p_y - (self.nu + nut)*(v_xx + v_yy)
        
        return u, v, p, f_c, f_u, f_v
    
    def callback(self, loss):
        print('Loss: %.3e' % (loss))
        self.fp.write('Loss: %.6e \n' % (loss))
        
    def get_batch(self,idx,bs,tb):
        
        return self.x[idx*bs:(idx+1)*bs],self.y[idx*bs:(idx+1)*bs],self.u[idx*bs:(idx+1)*bs],self.v[idx*bs:(idx+1)*bs]

      
    def train(self, nIter, lbfgs=False): 
        
        self.fp=open('conv.dat','w')
        
        total_batch=10
        batch_size=100
        lr=0.001
        min_lr=1e-6
        #reduce lr iter(patience)
        rli=300
        l_eps=1e-5
        #numbers to avg
        L=50
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
            
                batch_x, batch_y, batch_u, batch_v = self.get_batch(i,batch_size,total_batch)
                
                tf_dict = {self.x_tf: batch_x, self.y_tf: batch_y,
                   self.u_tf: batch_u, self.v_tf: batch_v, self.tf_lr:lr}
            
                _,loss_value=self.sess.run([self.train_op_Adam,self.loss], tf_dict)
                avg_loss += loss_value / 1
            
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
            print('It: %d, Loss: %.6e, lr:%0.6f, Time: %.2f' %(count, avg_loss,lr, elapsed))
            self.fp.write('It: %d, Loss: %.6e, lr:%0.6f, Time: %.2f \n' %(count, avg_loss,lr, elapsed))    
            start_time = time.time()
            
            #save model
            if ((count % 5000) ==0):
                model.save_model(count)
        
        #final_optimization using lbfgsb
        if (lbfgs==True):


                    
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y,
                       self.u_tf: self.u, self.v_tf: self.v}            
                
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
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
    xtmp=[]
    ytmp=[]
    reytmp=[]
    utmp=[]
    vtmp=[]
    ptmp=[]
    nuttmp=[]
    
    for ii in range(1):
        #x,y,Re,u,v
        with open('./data_file/cavity_Re20000.pkl', 'rb') as infile:
            result = pickle.load(infile)
        xtmp.extend(result[0])
        ytmp.extend(result[1])
        reytmp.extend(result[2])
        utmp.extend(result[3])
        vtmp.extend(result[4])
        ptmp.extend(result[5])   
        nuttmp.extend(result[6]) 
        
    xtmp=np.asarray(xtmp)
    ytmp=np.asarray(ytmp)
    utmp=np.asarray(utmp)
    vtmp=np.asarray(vtmp)
    ptmp=np.asarray(ptmp) 
    nuttmp=np.asarray(nuttmp)
            
    x = xtmp[:,None] # NT x 1
    y = ytmp[:,None] # NT x 1
    
    u = utmp[:,None] # NT x 1
    v = vtmp[:,None] # NT x 1
    p = ptmp[:,None] # NT x 1
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    N_train=1000
    idx = np.random.choice(len(xtmp), N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    # Training
    model = PhysicsInformedNN(x_train, y_train, u_train, v_train)
    
    model.train(50000,True)
    
    model.save_model(00000)
    
    # Prediction
    u_pred, v_pred, p_pred,_ = model.predict(xtmp[:,None], ytmp[:,None])
                                        

       
    plt.figure()
    plt.tricontourf(xtmp,ytmp,u_pred[:,0])
    plt.show()
    
    plt.figure()
    plt.tricontourf(xtmp,ytmp,utmp)
    plt.show()    

    
    
  

             
    


