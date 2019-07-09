"""
@author: Maziar Raissi
"""
'''
this is to make prediction using
p u v instead of original psi_p work
lamda fixed
Nu and Nf different points
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
    def __init__(self, x, y, u, v, layers, xf, yf):
        
        X = np.concatenate([x, y], 1)
        Xf = np.concatenate([xf, yf], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        self.Xf = Xf
        
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        
        self.xf=Xf[:,0:1]
        self.yf=Xf[:,1:2]
        
        self.u = u
        self.v = v
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        
        self.nu = tf.constant([0.01], dtype=tf.float32)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        
        self.xtr_f = tf.placeholder(tf.float32, shape=[None, self.xf.shape[1]])
        self.ytr_f = tf.placeholder(tf.float32, shape=[None, self.yf.shape[1]])        
        


        self.u_pred, self.v_pred, self.p_pred       = self.net_NS1(self.x_tf, self.y_tf)

        self.f_c_pred, self.f_u_pred, self.f_v_pred = self.net_NS2(self.xtr_f, self.ytr_f)

         
        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_sum(tf.square(self.f_c_pred)) + \
                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
                    tf.reduce_sum(tf.square(self.f_v_pred))

#        self.loss = tf.reduce_sum(tf.square(self.f_c_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
#                    tf.reduce_sum(tf.square(self.f_v_pred))

                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer(0.0001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

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
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS1(self, x, y):
        
        uvp = self.neural_net(tf.concat([x,y], 1), self.weights, self.biases)
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
             
        return u, v, p
    
    def net_NS2(self, x, y):
        lambda_1 = 1.0      
        lambda_2 = 1/100.
        
        uvp = self.neural_net(tf.concat([x,y], 1), self.weights, self.biases)
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
        
        return f_c, f_u, f_v
    
    def callback(self, loss):
        print('Loss: %.3e' % (loss))
      
    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y,
                   self.u_tf: self.u, self.v_tf: self.v,
                   self.xtr_f:self.xf, self.ytr_f:self.yf}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

        
    
    def predict(self, x_star, y_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, v_star, p_star

       
        
if __name__ == "__main__": 
      
        
    layers = [2, 30, 30, 30, 30, 30, 30, 30, 30, 30, 3]
    
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
        with open('./data_file_ldc/cavity_Re100.pkl', 'rb') as infile:
            result = pickle.load(infile,encoding='bytes')
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
           
#    x = xtmp[:,None] # NT x 1
#    y = ytmp[:,None] # NT x 1
#    
#    u = utmp[:,None] # NT x 1
#    v = vtmp[:,None] # NT x 1
#    p = ptmp[:,None] # NT x 1
    
    #boundary
    tmp=np.loadtxt('./data_file_ldc/ldc_bc.dat',skiprows=0)
    x=tmp[:,0:1]
    y=tmp[:,1:2]    
    u=tmp[:,2:3]
    v=tmp[:,3:4]
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    N_train=20
    idx = np.random.choice(len(x), N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    # Training Data    
    N_train=100
    idx = np.random.choice(len(xtmp), N_train, replace=False)
    x_train2 = xtmp[idx,None]
    y_train2 = ytmp[idx,None]
    u_train2 = utmp[idx,None]
    v_train2 = vtmp[idx,None]

    x_train = np.concatenate((x_train,x_train2),axis=0)
    y_train = np.concatenate((y_train,y_train2),axis=0)
    u_train = np.concatenate((u_train,u_train2),axis=0)
    v_train = np.concatenate((v_train,v_train2),axis=0)
    
    # Training Data    
    N_train_f=200
    idx = np.random.choice(len(xtmp), N_train_f, replace=False)
    x_train_f = xtmp[idx,None]
    y_train_f = ytmp[idx,None]



    # Training
    model = PhysicsInformedNN(x_train, y_train, u_train, v_train, layers, x_train_f, y_train_f)
    model.train(30000)
       
    # Prediction
    u_pred, v_pred, p_pred = model.predict(xtmp[:,None], ytmp[:,None])
                                        
    #save file
    filepath='./pred/ldc/'
    coord=[]  
    
    # ref:[x,y,z,ux,uy,uz,k,ep,nu
    info=['xtmp, ytmp, p, u, v, p_pred, u_pred, v_pred, x_train, y_train, info']

    data1 = [xtmp, ytmp, ptmp[:,None], u, v, p_pred, u_pred, v_pred, x_train, y_train, info]
    
    with open(filepath+'pred_ldc_rexxx.pkl', 'wb') as outfile1:
        pickle.dump(data1, outfile1, pickle.HIGHEST_PROTOCOL)
        
    plt.figure()
    plt.tricontourf(xtmp,ytmp,u_pred[:,0])
    plt.show()
    
    plt.figure()
    plt.tricontourf(xtmp,ytmp,utmp)
    plt.show()    

    
    
  

             
    


