from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model, model_from_json
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

import cPickle as pickle

path='./models/case_2_aae_uiuc_RS/saved_model/'
iters=99999

# load json and create model
json_file = open(path+'aae_decoder_%s_%s.json'%(iters,iters), 'r')
loaded_model_json = json_file.read()
json_file.close()
decoder = model_from_json(loaded_model_json)
# load weights into new model
decoder.load_weights(path+"aae_decoder_%s_weights_%s.hdf5"%(iters,iters))
print("Loaded model from disk")  


# load json and create model
json_file = open(path+'aae_encoder_%s_%s.json'%(iters,iters), 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_model_json)
# load weights into new model
encoder.load_weights(path+"aae_encoder_%s_weights_%s.hdf5"%(iters,iters))
print("Loaded model from disk")  

 # load json and create model
json_file = open(path+'aae_discriminator_%s_%s.json'%(iters,iters), 'r')
loaded_model_json = json_file.read()
json_file.close()
discriminator = model_from_json(loaded_model_json)
# load weights into new model
discriminator.load_weights(path+"aae_discriminator_%s_weights_%s.hdf5"%(iters,iters))
print("Loaded model from disk")  

#load data
myin=[]
myout=[]
xx=[]
name=[]

data_file='./data_file/foil_param_uiuc_216.pkl'	
with open(data_file, 'rb') as infile:
    result = pickle.load(infile)
    print (result[-1:]) 
    myin.extend(result[0])
    myout.extend(result[1])
    xx.extend(result[2])
    name.extend(result[3])
    
    myin=np.asarray(myin)    
    myin=np.reshape(myin,(len(myin),216,216,1))  
    myout=np.asarray(myout)/0.2
    xx=np.asarray(xx)
    
del result

#encoder
c1=encoder.predict(myin)
out=decoder.predict(c1)
score=discriminator.predict(c1)

eps=1e-12
mm=[]
for i in range(8):
    mm.append([c1[:,i].max()+eps,c1[:,i].min()+eps])
mm=np.asarray(mm)
    
mm_scale=[]    
for i in range(8):    
    mm_scale.append(max(abs(mm[i])))
mm_scale=np.asarray(mm_scale)

c1_scaled=c1.copy()
for i in range(8): 
    c1_scaled[:,i]=c1_scaled[:,i]/mm_scale[i]
    
info='[para_scaled,name,para(unscaled),mm_scaler,info:foil_param_uiuc_216.pkl]'    
data2=[c1_scaled,name,c1,mm_scale,info]
with open('./data_file/param_gan_uiuc_RS_8.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)
    
    
    
    
    

    
    
    
    
    
    
    
    
    
