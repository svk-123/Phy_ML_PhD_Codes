#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:02:28 2020

@author: vino
"""

import tensorflow as tf
import numpy as np
import scipy.optimize as sopt
'''
f(x)=x-(6/7)*x-1/7
g(x)=f(f(f(f(x))))
Find x such that g(x) == 0
'''


@tf.function
def rosen(a,b):
    return 100.0*(a-b**2.0)**2.0 + (1-b)**2.0

@tf.function
def model(a):
    return tf.math.reduce_sum(100.0*(a[1:]-a[:-1]**2.0)**2.0 + (1-a[:-1])**2.0)


@tf.function
def val_and_grad(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = model(x)
    grad = tape.gradient(loss, x)
    return loss, grad


#retun fucntion and gradients
def func(x):
    return [vv.numpy().astype(np.float64)  for vv in val_and_grad(tf.constant(x, dtype=tf.float32))]


#using scipy optimizer lbfgs
x0=np.asarray([0.5,0.5,0.5,0.5])
resdd= sopt.minimize(fun=func, x0=x0, jac=True, method='L-BFGS-B')

print("info:\n",resdd)
        
#using tfp optimizer lbfgs