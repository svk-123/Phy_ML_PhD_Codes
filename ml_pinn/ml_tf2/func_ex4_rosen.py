#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:02:28 2020

@author: vino
"""

import tensorflow as tf
import numpy as np

'''
f(x)=x-(6/7)*x-1/7
g(x)=f(f(f(f(x))))
Find x such that g(x) == 0
'''


x = tf.Variable([[0.5], [0.5], [0.5], [0.5]], trainable=True, dtype=tf.float64)

x1 = tf.Variable([0.5], trainable=True, dtype=tf.float64)
x2 = tf.Variable([0.5], trainable=True, dtype=tf.float64)

#c = tf.map_fn(lambda x: (x[0], x[1]), (x1,x2), dtype=(tf.float64,tf.float64))

y = tf.constant([0], dtype=tf.float64)

@tf.function
def rosen(a,b):
    return 100.0*(a-b**2.0)**2.0 + (1-b)**2.0

@tf.function
def rosen1(a):
    return tf.math.reduce_sum(100.0*(a[1:]-a[:-1]**2.0)**2.0 + (1-a[:-1])**2.0)

@tf.function
def add(a, b):
  return a + b

print(rosen1(x))

#print(tf.autograph.to_code(compute.python_function))

# Create a list of variables which needs to be adjusted during the training process, in this simple case it is only x
variables = [x]

# Instantiate a Gradient Decent Optimizer variant, it this case learning rate and specific type of optimizer doesn't matter too much
optimizer = tf.optimizers.Adam(0.001)

# We need to somehow specify the error between the actual value of the evaluated function in contrast to the target (which is zero)
loss_object = tf.keras.losses.MeanAbsoluteError()

# Since we are not running inside a TensorFlow execution graph anymore we need some means of keeping state of the gradient during training
# so a persistent GradientTape is your friend and the way to go in TensorFlow 2.0
with tf.GradientTape(persistent=True) as tape:
    
    #Let's train for some iterations
    y_pred=100.0
    iters=1
    while (y_pred > 1e-6):

        # given the actual value of X (which we now continueously adjust in order to find the root of the equation)
        y_pred = rosen1(x)

        # At this point we are actually setting the whole equation to zero. Since X is variable, the goal is to find an X which satisfies the condition
        # (that the whole equations becomes zero). We are doing this by defining a loss which becomes zero if y_pred approximates y. Or in other words,
        # since y is zero, the loss becomes zero if y_pred approximates zero.
        loss = loss_object(y,y_pred)

        # Now the magic happens. Loss basically represents the error surface and is only dependent on X. So now let's compute the first derivative and
        # see in which direction we need to adjust X in order to minimize the error and getting a value (output of the nested equations) closer to zero
        grads = tape.gradient(loss, variables)

        # Once we've found this magic number magically, let's update the value of X based on this magic number in order to perform better on the next
        # iteration
        optimizer.apply_gradients(zip(grads, variables))

        # And now it's pretty cool, we can just print the current error (loss) and the actual value of X in each iteration. At the end of the training,
        # we've found the optima wich a loss / error close to zero and a value of X close to 400 where 400 is the correct solution.
        # Small deviations from the true solutions stem from numeric errors
        print('Iters: {}, Loss: {}, X: {}'.format(iters,loss.numpy(), x.numpy()))
        iters=iters+1
        
