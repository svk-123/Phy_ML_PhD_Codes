from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

import cPickle as pickle

class AdversarialAutoencoder():
    def __init__(self):

        self.img_shape = (216,216,1)
        self.latent_dim = 8
        self.my_lr=0.001
        optimizer = Adam(self.my_lr)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)


    def build_encoder(self):
        # Encoder

        img= Input(shape=self.img_shape)

        # 2 3x3 convolutions followed by a max pool
        conv1 = Conv2D(64, (8, 8), activation='relu', padding='same')(img)
        #conv1 = Conv2D(32, (4, 4), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(3, 3))(conv1)
        
        conv2 = Conv2D(128, (6, 6), activation='relu', padding='same')(pool1)
        #conv2 = Conv2D(64, (4, 4), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(3, 3))(conv2)
        
        conv3 = Conv2D(128, (4, 4), activation='relu', padding='same')(pool2)
        pool3 = MaxPooling2D(pool_size=(3, 3))(conv3)
        
        conv4 = Conv2D(256, (2, 2), activation='relu', padding='same')(pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        #conv5 = Conv2D(128, (2, 2), activation='relu', padding='same')(pool4)
        #pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
        
        # flatten the 4D array (batch, height, width, depth) into 
        # a 2D array (batch, n). Perform a fully connected layer
        flat5 = Flatten()(pool4)
        
        flat5 = Dense(200, activation='relu')(flat5)
        flat5 = Dense(200, activation='relu')(flat5)
        flat5 = Dense(200, activation='relu')(flat5)
        latent_repr = Dense(self.latent_dim, activation='sigmoid')(flat5)
                    
        return Model(img, latent_repr)

    def build_decoder(self):

        model = Sequential()

        model.add(Dense(200, activation='relu', input_dim=self.latent_dim))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(100,  activation='linear'))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        img = model(z)

        return Model(z, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(100, activation='tanh', input_dim=self.latent_dim))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(1,   activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    def train(self, epochs, batch_size=32, sample_interval=50):

        # Load the dataset
        myin=[]
        myout = []
        xx  = []

        data_file='../data_file/foil_param_uiuc_216.pkl'	
        with open(data_file, 'rb') as infile:
            result = pickle.load(infile)
            print (result[-1:])    
            
            myin.extend(result[0])
            myout.extend(result[1])
            xx.extend(result[2])
        
        myin=np.asarray(myin)
        myin=np.reshape(myin,(len(myin),216,216,1))  
        
        myout=np.asarray(myout)/0.2

        self.xx=np.asarray(xx)

        # Rescale -1 to 1
        X_train = myin
        Y_train = myout
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            out  = Y_train[idx]
            
            latent_real = self.encoder.predict(imgs)
            #latent_fake = np.random.normal(size=(batch_size, self.latent_dim))
            latent_fake=np.random.random_sample((batch_size,self.latent_dim))
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [out, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f], lr=%e" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1],\
                   self.my_lr))

            # If at save interval => save generated image samples
            #if epoch % sample_interval == 0:
                #self.sample_images(epoch)
                #self.training_images(epoch)
                
            if ((epoch+1) % 1000 == 0):
                self.my_lr=self.my_lr*0.5
                
        self.save_model()
        
    def sample_images(self, epoch):
        r, c = 5, 5

        #z = np.random.normal(size=(r*c, self.latent_dim))
        z= np.random.random_sample((25,self.latent_dim))
        gen_imgs = self.decoder.predict(z)
        score_ =     self.discriminator(K.constant(z))
        score=K.get_value(score_)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].plot(self.xx[::-1],gen_imgs[cnt, :35])
                axs[i,j].plot(self.xx,gen_imgs[cnt, 35:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()
        
        fp=open('images/score_%d.dat'%epoch,'w')
        for i in range(len(score)):
            fp.write('%s \n'%score[i])
        fp.close()  
        
    def training_images(self, epoch):
        
        out=[]
        xx=[]

        data_file='../data_file/foil_param_uiuc_216.pkl'	
        with open(data_file, 'rb') as infile:
            result = pickle.load(infile)
            print (result[-1:])    
            out.extend(result[1])
            xx.extend(result[2])
        
        out=np.asarray(out)/0.25
        self.xx=np.asarray(xx)
        del result
        
        r, c = 5, 5

        z = self.encoder.predict(out[:25])
        gen_imgs = self.decoder.predict(z)
        score=     self.discriminator.predict(z)
        print (gen_imgs.shape)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].plot(self.xx[::-1],gen_imgs[cnt, :35])
                axs[i,j].plot(self.xx,gen_imgs[cnt, 35:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("true_images/mnist_%d.png" % epoch)
        plt.close()
        
        fp=open('true_images/score_%d.dat'%epoch,'w')
        for i in range(len(score)):
            fp.write('%s \n'%score[i])
        fp.close()             
        
    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.adversarial_autoencoder, "aae_generator")
        save(self.discriminator, "aae_discriminator")
        save(self.encoder, "aae_encoder")
        save(self.decoder, "aae_decoder")


if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=10000, batch_size=32, sample_interval=100)
