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
    def __init__(self,x,y,u,v,p):

        self.x = x
        self.y = y
        
        self.u = u
        self.v = v
        self.p = p                              
   
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1],name='ipt0')
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1],name='ipt1')
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.p_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        self.nu = tf.constant([0.01], dtype=tf.float32)
        
        self.u_pred, self.v_pred, self.p_pred, self.f_c_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf)
        
        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_sum(tf.square(self.p_tf - self.p_pred)) + \
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

    
    def neural_net(self, X):

        #create model
        l1 = tf.layers.dense(X,  30, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 30, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 30, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 30, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 30, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 30, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 30, activation=tf.nn.tanh)
        Y  = tf.layers.dense(l1,3,activation=None,name='prediction')
        
        
        return Y
        
    def net_NS(self, x, y):

        uvp = self.neural_net(tf.concat([x,y], 1))
        
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
        f_u =  (u*u_x + v*u_y) + p_x - self.nu*(u_xx + u_yy) 
        f_v =  (u*v_x + v*v_y) + p_y - self.nu*(v_xx + v_yy)
        
        return u, v, p, f_c, f_u, f_v
    
    def callback(self, loss):
        print('Loss: %.3e' % (loss))
        
    def get_batch(self,idx,bs,tb):
        return self.x[idx*bs:(idx+1)*bs],self.y[idx*bs:(idx+1)*bs],self.u[idx*bs:(idx+1)*bs], \
                self.v[idx*bs:(idx+1)*bs],self.p[idx*bs:(idx+1)*bs]
      
    def train(self, nIter, lbfgs=False): 
        
        fp=open('conv.dat','w')
        

        batch_size=50
          
        total_batch=20
        
        lr=0.0001
        min_lr=1e-6
        #reduce lr iter(patience)
        rli=200
        #numbers to avg
        L=30
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
            
                batch_x, batch_y, batch_u, batch_v, batch_p = self.get_batch(i,batch_size,total_batch)
                
                tf_dict = {self.x_tf: batch_x, self.y_tf: batch_y,
                   self.u_tf: batch_u, self.v_tf: batch_v, self.p_tf: batch_p, self.tf_lr:lr}
            
                _,loss_value=self.sess.run([self.train_op_Adam,self.loss], tf_dict)
                avg_loss += loss_value / total_batch
            
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
            print('It: %d, Loss: %.6e, lr:%0.6f, Time: %.2f' %(count, avg_loss,lr, elapsed))
            fp.write('It: %d, Loss: %.6e, lr:%0.6f, Time: %.2f \n' %(count, avg_loss,lr, elapsed))    
            start_time = time.time()
            
       
        #final_optimization using lbfgsb
        if (lbfgs==True):
                    
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y,
                       self.u_tf: self.u, self.v_tf: self.v, self.p_tf: self.p}            
                
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
    ytmp=[]
    reytmp=[]
    utmp=[]
    vtmp=[]
    ptmp=[]
    
    for ii in range(1):
        #x,y,Re,u,v
        with open('./data_file/naca0006_100_0_part.pkl', 'rb') as infile:
            result = pickle.load(infile,encoding='bytes')
        xtmp.extend(result[0])
        ytmp.extend(result[1])
        ptmp.extend(result[2])
        utmp.extend(result[3])
        vtmp.extend(result[4])   
        
    xtmp=np.asarray(xtmp)
    ytmp=np.asarray(ytmp)
    utmp=np.asarray(utmp)
    vtmp=np.asarray(vtmp)
    ptmp=np.asarray(ptmp) 
           
    x = xtmp[:,None] # NT x 1
    y = ytmp[:,None] # NT x 1
    
    u = utmp[:,None] # NT x 1
    v = vtmp[:,None] # NT x 1
    p = ptmp[:,None] # NT x 1
    
    #load BC data
    xy_bc=np.loadtxt('./data_file/naca0006.dat',skiprows=1)
    x_bc=xy_bc[:,0:1]
    y_bc=xy_bc[:,1:2]
    u_bc=x_bc.copy()
    v_bc=y_bc.copy()
    u_bc[:,:] = 0
    v_bc[:,:] = 0
          
        
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
    p_train = p[idx,:]
    
#    #concate with bc points
#    x_train = np.concatenate((x_train,x_bc),axis=0)
#    y_train = np.concatenate((y_train,y_bc),axis=0)
#    u_train = np.concatenate((u_train,u_bc),axis=0)
#    v_train = np.concatenate((v_train,v_bc),axis=0)
#    
#    #shuffle
#    idx = np.random.choice(len(x_train), len(x_train), replace=False)
#    x_train = x[idx,:]
#    y_train = y[idx,:]
#    u_train = u[idx,:]
#    v_train = v[idx,:]

    # Training
    model = PhysicsInformedNN(x_train, y_train, u_train, v_train, p_train)
    
    model.train(100000)
    
    model.save_model()
    
    # Prediction
    u_pred, v_pred, p_pred = model.predict(xtmp[:,None], ytmp[:,None])
                                        
    #save file
    filepath='./pred/airfoil/'
    coord=[]  
    # ref:[x,y,z,ux,uy,uz,k,ep,nu
    info=['xtmp, ytmp, p, u, v, p_pred, u_pred, v_pred, x_train, y_train, info']

    data1 = [xtmp, ytmp, p, u, v, p_pred, u_pred, v_pred, x_train, y_train, info]
    
    with open(filepath+'pred_naca0006_re100.pkl', 'wb') as outfile1:
        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)
        
    plt.figure()
    plt.tricontourf(xtmp,ytmp,u_pred[:,0])
    plt.show()
    
    plt.figure()
    plt.tricontourf(xtmp,ytmp,utmp)
    plt.show()    

    
    
  

             
    


