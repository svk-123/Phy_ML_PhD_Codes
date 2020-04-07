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

# load json and create model
json_file = open('./saved_model/aae_decoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
decoder = model_from_json(loaded_model_json)
# load weights into new model
decoder.load_weights("./saved_model/aae_decoder_weights.hdf5")
print("Loaded model from disk")  


# load json and create model
json_file = open('./saved_model/aae_encoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_model_json)
# load weights into new model
encoder.load_weights("./saved_model/aae_encoder_weights.hdf5")
print("Loaded model from disk")  

 # load json and create model
json_file = open('./saved_model/aae_discriminator.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
discriminator = model_from_json(loaded_model_json)
# load weights into new model
discriminator.load_weights("./saved_model/aae_discriminator_weights.hdf5")
print("Loaded model from disk")  

#load data
out=[]
xx=[]

data_file='foil_param_216_no_aug_tr.pkl'	
with open(data_file, 'rb') as infile:
    result = pickle.load(infile)
    print (result[-1:])    
    out.extend(result[1])
    xx.extend(result[2])
        
    out=np.asarray(out)/0.25
    xx=np.asarray(xx)
    
del result

#encoder
latent=encoder.predict(out[500:1000])
myout=decoder.predict(latent)
#score=discriminator.predict(latent)


#cnt = 0
#for k in range(4):
#        r=5
#        c=5
#        fig, axs = plt.subplots(r, c)
#        
#        for i in range(r):
#            for j in range(c):
#                axs[i,j].plot(xx[::-1],myout[cnt, :35])
#                axs[i,j].plot(xx,myout[cnt, 35:])
#                axs[i,j].axis('off')
#                cnt += 1
#        fig.savefig("plot/true_%d.png" % k)
#        plt.close()   



    
for k in range(100):
        I= np.random.randint(0,100)
        l=latent[I]
        noise=abs(np.random.normal(0,0.1,10))
        out1=decoder.predict((l[:,None]+noise[:,None]).transpose())
        score1=discriminator.predict((l[:,None]+noise[:,None]).transpose())
        
        #true
        out2=decoder.predict((l[:,None]).transpose())    
        score2=discriminator.predict((l[:,None]).transpose())
        
        plt.figure()
        plt.plot(xx[::-1],out1[0, :35],'r')
        plt.plot(xx,out1[0, 35:],'r')
        plt.plot(xx[::-1],out2[0, :35],'g')
        plt.plot(xx,out2[0, 35:],'g')
        #plt.axis('off')
        plt.ylim([-1,1])
        plt.text(0.4,-0.9,'%s'%score1)
        plt.savefig("plot/true_%d.png" % k)
        
        plt.show()
        plt.close()     





##random sample    
#cnt = 0
#score=[]
#for k in range(1000):
#    latent_fake=np.random.random_sample((1,10))
#    score.append(discriminator.predict(latent_fake))
#score=np.asarray(score)
    
    

    
    
    
    
    
    
    
    
    