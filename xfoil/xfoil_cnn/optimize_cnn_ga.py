"""
Optimize the airfoil shape directly using genetic algorithm, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)

Reference(s):
    Viswanath, A., J. Forrester, A. I., Keane, A. J. (2011). Dimension Reduction for Aerodynamic Design Optimization.
    AIAA Journal, 49(6), 1256-1266.
    Grey, Z. J., Constantine, P. G. (2018). Active subspaces of airfoil shape parameterizations.
    AIAA Journal, 56(5), 2003-2017.
"""

from __future__ import division
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from genetic_alg import generate_first_population, select, create_children, mutate_population
from cnn.synthesis import synthesize
from utils import mean_err

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers import merge, Input, dot
from sklearn.metrics import mean_squared_error
import random

from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.callbacks import TensorBoard
import pickle

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K


def optimize(x0, syn_func, perturb_type, perturb, n_eval, run_id):
    # Optimize using GA
    n_best = 30
    n_random = 10
    n_children = 5
    chance_of_mutation = 0.1
    population_size = int((n_best+n_random)/2*n_children)
    population = generate_first_population(x0, population_size, perturb_type, perturb)
    best_inds = []
    best_perfs = []
    opt_perfs = [0]
    i = 0
    while 1:
        breeders, best_perf, best_individual = select(population, n_best, n_random, syn_func)
        best_inds.append(best_individual)
        best_perfs.append(best_perf)
        opt_perfs += [np.max(best_perfs)] * population_size # Best performance so far
        print('PARSEC-GA %d-%d: fittest %.2f' % (run_id, i+1, best_perf))
        # No need to create next generation for the last generation
        if i < n_eval/population_size-1:
            next_generation = create_children(breeders, n_children)
            population = mutate_population(next_generation, chance_of_mutation, perturb_type, perturb)
            i += 1
        else:
            break
    
    opt_x = best_inds[np.argmax(best_perfs)]
    opt_airfoil = synthesize(opt_x, n_points)
    print('Optimal CL/CD: {}'.format(opt_perfs[-1]))
    
    return opt_airfoil, opt_perfs


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Optimize')
    parser.add_argument('--n_runs', type=int, default=2, help='number of runs')
    parser.add_argument('--n_eval', type=int, default=10, help='number of evaluations per run')
    args = parser.parse_args()
    
    n_runs = args.n_runs
    n_eval = args.n_eval
    
    # Airfoil parameters
    n_points = 192
    
    #############################-----------------------
    cd=[]
    cl=[]
    mypara=[]
    name=[]
    xx=[]
    
    #load airfoil para
    path='./model_cnn_v1/'
    data_file='cnn_uiuc_para_8_tanh_v1.pkl'
    with open(path + data_file, 'rb') as infile:
        result2 = pickle.load(infile)
    
    
    mypara.extend(result2[0])
    name.extend(result2[1])
    xx.extend(result2[2])
    
    mypara=np.asarray(mypara)
    name=np.asarray(name)
    xx=np.asarray(xx)
    xx=xx[:101] 
    
    model_para=load_model('./model_cnn_v1/case_1_p8_tanh_include_naca4_v2/model_cnn/final_cnn.hdf5') 
    get_c= K.function([model_para.layers[16].input],  [model_para.layers[19].output])    
    
    #base foil name
    idx1=np.random.randint(1200)   
    #fn=name[idx1]
    fn='n0012'
    print(fn)
    idx=np.argwhere(name=='%s'%fn) 
    
    # NACA 0012 as the original airfoil
    x0 = mypara[idx[0][0],:]
    
    perturb_type = 'absolute'
    perturb = 0.05
    syn_func = lambda x: synthesize(x, n_points)
    
    opt_airfoil_runs = []
    opt_perfs_runs = []
    time_runs = []
    for i in range(n_runs):
        start_time = time.time()
        opt_airfoil, opt_perfs = optimize(x0, syn_func, perturb_type, perturb, n_eval, i+1)
        end_time = time.time()
        opt_airfoil_runs.append(opt_airfoil)
        opt_perfs_runs.append(opt_perfs)
        time_runs.append(end_time-start_time)
    
    opt_airfoil_runs = np.array(opt_airfoil_runs)
    opt_perfs_runs = np.array(opt_perfs_runs)
    np.save('opt_results/parsec_ga/opt_airfoil.npy', opt_airfoil_runs)
    np.save('opt_results/parsec_ga/opt_history.npy', opt_perfs_runs)
    
    # Plot optimization history
    mean_perfs_runs = np.mean(opt_perfs_runs, axis=0)
    plt.figure()
    plt.plot(np.arange(n_eval+1, dtype=int), opt_perfs)
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Optimal CL/CD')
#    plt.xticks(np.linspace(0, n_eval+1, 5, dtype=int))
    plt.savefig('opt_results/parsec_ga/opt_history.svg')
    plt.close()
    
    # Plot the optimal airfoil
    mean_time_runs, err_time_runs = mean_err(time_runs)
    mean_final_perf_runs, err_final_perf_runs = mean_err(opt_perfs_runs[:,-1])
    plt.figure()
    for opt_airfoil in opt_airfoil_runs:
        plt.plot(opt_airfoil[:,0], opt_airfoil[:,1], '-', c='k', alpha=1.0/n_runs)
    plt.title('CL/CD: %.2f+/-%.2f  time: %.2f+/-%.2f min' % (mean_final_perf_runs, err_final_perf_runs, 
                                                             mean_time_runs/60, err_time_runs/60))
    plt.axis('equal')
    plt.savefig('opt_results/parsec_ga/opt_airfoil.svg')
    plt.close()

    print 'PARSEC-GA completed :)'
