#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 22:11:57 2020

@author: vino
"""

import tensorflow as tf
import numpy as np
import scipy.optimize as sopt


x = tf.Variable([[0.5,0.5]], trainable=True, dtype=tf.float64)
y = tf.constant([0], dtype=tf.float64)

#load rosen mlp model
model_mlp= tf.keras.models.load_model('rosen_hr_mlp')
def model(a):
    return model_mlp(a)

def val_and_grad(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = model(x)
    grad = tape.gradient(loss, x)
    return loss, grad

#retun fucntion and gradients
def func(x):
    return [vv.numpy().astype(np.float64)  for vv in val_and_grad(tf.Variable([x], dtype=tf.float32))]

#using scipy optimizer lbfgs
x0=np.asarray([0.0,0.0])
resdd= sopt.minimize(fun=func, x0=x0, jac=True, method='L-BFGS-B')

print("info:\n",resdd)