#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 01:25:50 2020

@author: vino
"""

import tensorflow as tf

x = tf.Variable(2, name='x', trainable=True, dtype=tf.float32)
with tf.GradientTape() as t:
    # no need to watch a variable:
    # trainable variables are always watched
    log_x = tf.math.log(x)
    y = tf.math.square(log_x)

opt = tf.optimizers.Adam(learning_rate=0.001)

#### Option 1

# Is the tape that computes the gradients!
trainable_variables = [x]
gradients = t.gradient(y, trainable_variables)
# The optimize applies the update, using the variables
# and the optimizer update rule
opt.apply_gradients(zip(gradients, trainable_variables))