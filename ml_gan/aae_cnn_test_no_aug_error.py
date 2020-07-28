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
from numpy import linalg as LA
import matplotlib.pyplot as plt

import numpy as np

import cPickle as pickle

path='./models/case_1_aae_no_aug_RT/saved_model/'
iters=99000

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

data_file='./data_file/foil_param_216_no_aug_ts.pkl'	
with open(data_file, 'rb') as infile:
    result = pickle.load(infile)
    print (result[-1:]) 
    myin.extend(result[0])
    myout.extend(result[1])
    xx.extend(result[2])
    
    myin=np.asarray(myin)    
    myin=np.reshape(myin,(len(myin),216,216,1))  
    myout=np.asarray(myout)
    xx=np.asarray(xx)
del result

#encoder
latent=encoder.predict(myin)
out=decoder.predict(latent)
out=out*0.25
score=discriminator.predict(latent)


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


# morphed airfoils
cnt=0    
for k in range(100):
        
        I= np.random.randint(0,100)
        l=latent[I]
        noise=np.random.normal(0,0.1,8)
        out1=decoder.predict((l[:,None]+noise[:,None]).transpose())
        out1=out1*0.25
        score1=discriminator.predict((l[:,None]+noise[:,None]).transpose())
        print (score1[0][0])
        if(score1[0][0] > 0.999999):
            #true
            print (cnt)
            out2=decoder.predict((l[:,None]).transpose())    
            out2=out2*0.25
            score2=discriminator.predict((l[:,None]).transpose())
            
            #plot
            plt.figure()
            plt.plot(xx[::-1],out1[0,:35],'r')
            plt.plot(xx,out1[0,35:],'r')
            plt.plot(xx[::-1],out2[0,:35],'g')
            plt.plot(xx,out2[0,35:],'g')
            #plt.axis('off')
            plt.ylim([-0.2,0.2])
            #plt.text(0.4,-0.9,'%s'%score1)
            plt.savefig("plot/true_%d.png" % k)
            
            plt.show()
            plt.close()     
            cnt=cnt+1


            
###random sample    
##cnt = 0
##score=[]
##for k in range(1000):
##    latent_fake=np.random.random_sample((1,10))
##    score.append(discriminator.predict(latent_fake))
##score=np.asarray(score)
#    
#    
##calculate error norm
#train_l2=[]
#train_l1=[]
#for k in range(len(out)):    
#    
#    tmp=myout[k]-out[k]
#    
#    train_l2.append( (LA.norm(tmp)/LA.norm(out))*100 )
#
#    tmp2=tmp/out[k]
#    train_l1.append(sum(abs(tmp2))/len(out))
#print ("train_l2_avg",sum(train_l2)/len(train_l2))
##spread_plot
#plt.figure(figsize=(6,5),dpi=100)
#plt.plot([-2,1.5],[-2,1.5],'k',lw=3)
#plt.plot(myout[0],out[0],'ro')
#for k in range(len(out)):
#    
#    plt.plot(myout[k],out[k],'ro')
#plt.legend(fontsize=20)
#plt.xlabel('True',fontsize=20)
#plt.ylabel('Prediction',fontsize=20)
##lt.xlim([-1.5,1.5])
##plt.ylim([-1.5,1.5])    
#plt.savefig('test_spread.png', bbox_inches='tight',dpi=100)
#plt.show() 
#    
##error plot
#plt.figure(figsize=(6,5),dpi=100)
#plt.hist(train_l2, 20,histtype='step', color='grey',stacked=True,fill=True,alpha=1,orientation ='vertical')
#plt.ylabel('Number of Samples',fontsize=20)
#plt.xlabel('$L_2$ relative error(%)',fontsize=20)
#plt.figtext(0.40, 0.01, '(b)', wrap=True, horizontalalignment='center', fontsize=24)    
#plt.subplots_adjust(top = 0.95, bottom = 0.22, right = 0.9, left = 0, hspace = 0, wspace = 0.1)
##plt.xlim([-0.0,0.25])
##plt.xticks([0,0.5,1.])
#plt.savefig('tr_tot.tiff',format='tiff', bbox_inches='tight',dpi=300)
#plt.show()
#
#    
#    
#
##testing airfoils
#cnt=0    
#for k in range(len(out)):
#                
#        if(2 > 1):
#            #true
#            print (cnt)
#            
#            #plot
#            plt.figure()
#            plt.plot(xx[::-1],myout[k,0:35],'r')
#            plt.plot(xx,myout[k,35:],'r')
#            
#            plt.plot(xx[::-1],out[k,0:35],'g')
#            plt.plot(xx,out[k,35:],'g')            
#
#
#            #plt.axis('off')
#            plt.ylim([-0.25,0.25])
#            plt.text(0.4,-0.9,'%s'%score[k])
#            plt.savefig("plot/ts_%d.png" % k)
#            
#            plt.show()
#            plt.close()     
#            cnt=cnt+1    
#    
#    
#    
#    