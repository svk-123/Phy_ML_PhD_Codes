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
from matplotlib import cm

np.random.seed(12734)
tf.set_random_seed(17234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self,x,y,u):

        self.x = x
        self.y = y
        
        self.u = u

                              
   
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1],name='ipt0')
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1],name='ipt1')
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        self.y_pred = self.net_NS(self.x_tf, self.y_tf)
        
        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.y_pred)) 

                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 10000,
                                                                           'maxfun': 10000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer(self.tf_lr)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        
        self.sess.run(init)

    
    def neural_net(self, X):

        #create model
        l1 = tf.layers.dense(X,  10, activation=tf.nn.tanh)
        l1 = tf.layers.dense(l1, 10, activation=tf.nn.tanh)
        Y  = tf.layers.dense(l1,1,activation=None,name='prediction')
        
        
        return Y
        
    def net_NS(self, x, y):
        
        uvp = self.neural_net(tf.concat([x,y], 1))
        
        u = uvp[:,0:1]
        
        return u
    
    def callback(self, loss):
        print('Loss: %.3e' % (loss))
        
    def get_batch(self,idx,bs,tb):
        
        if(idx<tb):
            return self.x[idx*bs:(idx+1)*bs],self.y[idx*bs:(idx+1)*bs],self.u[idx*bs:(idx+1)*bs],self.v[idx*bs:(idx+1)*bs]
        else:
            return self.x[(idx+1)*bs:],self.y[(idx+1)*bs:],self.u[(idx+1)*bs:],self.v[(idx+1)*bs:]
      
    def train(self, nIter, lbfgs=False): 
        
        fp=open('conv.dat','w')
        
        total_batch=1
        batch_size=1
        
        lr=0.001
        min_lr=1e-6
        #reduce lr iter(patience)
        rli=200
        #numbers to avg
        L=20
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
                
                tf_dict = {self.x_tf: self.x, self.y_tf: self.y,
                   self.u_tf: self.u, self.tf_lr:lr}
            
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
            print('It: %d, Loss: %.3e, Time: %.2f' %(count, avg_loss, elapsed))
            fp.write('It: %d, Loss: %.3e, Time: %.2f \n' %(count, avg_loss, elapsed))    
            start_time = time.time()
            
        
        #final_optimization using lbfgsb
        if (lbfgs==True):


                    
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y,
                       self.u_tf: self.u}            
                
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)
 

        fp.close()
    
    def predict(self, x_star, y_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}
        
        u_star = self.sess.run(self.y_pred, tf_dict)

        
        return u_star

    def save_model(self):
        
        saver = tf.train.Saver()
        saver.save(self.sess,'./tf_model/model') 
        
        
if __name__ == "__main__": 
      
            
    aa=np.linspace(-5,10,100)
    bb=np.linspace(0,15,100)

    X1,X2 = np.meshgrid(aa,bb,indexing='ij')
    Y=X1.copy()
    Y[:,:]=0
    


    a=1
    b=5.1/(4*np.pi**2)
    c=5/np.pi
    r=6
    s=10
    t=1/(8*np.pi)
    for i in range(100):
        for j in range(100):
            x1=X1[i,j]
            x2=X2[i,j]
            
            y =  a * (x2 - (b*x1**2) + (c*x1) - r)**2 + \
            s*(1-t)*np.cos(x1) + s
            
            Y[i,j] = y
                       
     
    X1=X1/10
    X2=X2/15
    Y=Y/308.12909601160663

    x1=X1.flatten()[:,None]
    x2=X2.flatten()[:,None]
    y=Y.flatten()[:,None]
    
    xy=np.loadtxt('branin_lit.dat',delimiter=',')
    x11=xy[:,0]
    x22=xy[:,1]
    y11=np.zeros(20)
    for j in range(1):
        for i in range(20):
            
            y11[i] =  a * (x22[i] - (b*x11[i]**2) + (c*x11[i]) - r)**2 + \
            s*(1-t)*np.cos(x11[i]) + s
            
    x11=x11/10
    x22=x22/15
    y11=y11/308.12909601160663
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    N_train=20
    idx = np.random.choice(10000, N_train, replace=False)
    x_train = x1[idx,:]
    y_train = x2[idx,:]
    u_train = y[idx,:]

    # Training
    #model = PhysicsInformedNN(x_train, y_train, u_train)
    
    model = PhysicsInformedNN(x11[:,None], x22[:,None], y11[:,None])    
    model.train(50000)
    
    model.save_model()
    
    # Prediction
    y_pred = model.predict(x1,x2)
    
    plt.figure(figsize=(6, 5), dpi=100)                                            
    cp = plt.tricontour(x1[:,0],x2[:,0],y[:,0],20,linewidths=1,cmap=cm.jet)
    plt.plot(x11,x22,'ok')           
    plt.xlim(-0.5,1)
    plt.ylim(0,1)
    plt.colorbar()
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    #plt.figtext(0.5, 0.01, '%s'%sub_name, wrap=True, horizontalalignment='center', fontsize=24)
    #plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.98, left = 0.14, hspace = 0, wspace = 0)
    plt.savefig('./plot/true.tiff',format='tiff' , bbox_inch='tight', dpi=300)
    plt.show()
    plt.close()
    
    plt.figure(figsize=(6, 5), dpi=100)                                            
    cp = plt.tricontour(x1[:,0],x2[:,0],y_pred[:,0],20,linewidths=1,cmap=cm.jet)
    plt.plot(x11,x22,'ok')             
    plt.xlim(-0.5,1)
    plt.ylim(0,1)
    plt.colorbar()
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    #plt.figtext(0.5, 0.01, '%s'%sub_name, wrap=True, horizontalalignment='center', fontsize=24)
    #plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.98, left = 0.14, hspace = 0, wspace = 0)
    plt.savefig('./plot/pred.tiff',format='tiff' , bbox_inch='tight', dpi=300)
    plt.show()
    plt.close()   




    
    tmp1=abs(y[:,0]-y_pred[:,0])*308.12909601160663
    tmp2=tmp1**2
    
    print ('MSE',sum(tmp2))

             
    


