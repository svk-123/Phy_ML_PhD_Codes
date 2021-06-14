#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 03:02:03 2020

@author: vino
"""

from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

import matplotlib.pyplot as plt

num = 20
images = X_train[:num]
labels = Y_train[:num]

# pick a sample to plot
sample = 1
image = X_train[sample]
# plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray_r')
plt.show()

num_row = 4
num_col = 5# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
plt.tight_layout()
plt.savefig('mnist_g.png',format='png',dpi=300)
plt.show()

img=image
fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        #val=val/255.0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
     
plt.savefig('mnist_digit.png',format='png',dpi=300)
        