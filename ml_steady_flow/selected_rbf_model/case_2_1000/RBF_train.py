# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
#import matplotlib.tri as tri
from matplotlib import cm
import time
import pickle
#import os
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * np.linalg.norm(x-c)**2)

""" function: load_data """
def load_data(fileDir, ReNo, fileExt):
    
    fileFullDir = fileDir + str(ReNo) + fileExt

    with open(fileFullDir, "rb") as fid:
        data = pickle.load(fid)

    return data
""" function: load_data """

""" function: flow field plot """
def plot(xp,yp,zp,nc,name):

    plt.figure(figsize=(6, 5), dpi=100)
    #cp = pyplot.tricontour(ys, zs, pp,nc)
    cp = plt.tricontourf(xp,yp,zp,nc,cmap=cm.jet)
#    v= np.linspace(0, 0.05, 15, endpoint=True)
    #cp = plt.tricontourf(xp,yp,zp,v,cmap=cm.jet,extend='both')
    #cp = pyplot.tripcolor(ys, zs, pp)
    #cp = pyplot.scatter(ys, zs, pp)
    #pyplot.clabel(cp, inline=False,fontsize=8)
    plt.colorbar()
    #plt.title('%s  '%flist[ii]+name)
    plt.xlabel('X ',fontsize=20)
    plt.ylabel('Y ',fontsize=20)
    plt.savefig("./Result_pics/"+name+".png", dpi=100)
#    plt.show()
""" function: flow field plot """


""" main function """
if __name__ == "__main__":

    # load flow field datasets from data directory
    # data: x, y, ReNo, u, v, p
    
    ReNos = np.array([100,200,400,600,1000,2000,3000,4000,5000,7000,8000,9000])
    inputNo = 3
    outputNo = 3
    noNeurons = 1000
    
    inputSamples = np.zeros((1,inputNo))
    outputSamples = np.zeros((1,outputNo))

    for i in range(len(ReNos)):
        data = load_data("../data/cavity_Re", ReNos[i], ".pkl")
        sampleNo = len(data[0])
        # input
        Temp = np.zeros((sampleNo, inputNo))
        for i in range(inputNo):
            for j in range(sampleNo):
                Temp[j,i] = data[i][j]
        inputSamples = np.concatenate((inputSamples,Temp), axis=0)

        # output
        Temp = np.zeros((sampleNo, outputNo))
        for i in range(3, 6):
            for j in range(sampleNo):
                Temp[j, i-3] = data[i][j]
        outputSamples = np.concatenate((outputSamples,Temp), axis=0)
    
    inputSamples = np.delete(inputSamples,0,axis=0)
    outputSamples = np.delete(outputSamples,0,axis=0)

    sampleNo = inputSamples.shape[0]
    # normalize - input
    oriInput = inputSamples.copy()
    
    iMin = np.array([0,0,ReNos[0]])
    iMax = np.array([1,1,ReNos[-1]])
    iDiff = iMax - iMin
    
    for i in range(sampleNo):
        for j in range(inputNo):
            inputSamples[i, j] = (inputSamples[i, j] -iMin[j])/iDiff[j]
            
    # normalize - output
    oriOutput = outputSamples.copy()
            
    oMin = np.amin(outputSamples, axis=0)
    oMax = np.amax(outputSamples, axis=0)
    oDiff = oMax - oMin
    
    for i in range(sampleNo):
        for j in range(outputNo):
            outputSamples[i, j] = (outputSamples[i, j] -oMin[j])/oDiff[j]
    
    #shuffle data
    N= len(inputSamples)
    I = np.arange(N)
    np.random.shuffle(I)
    n=N

    ## Training sets
    inputSamples= inputSamples[I][:n]
    outputSamples= outputSamples[I][:n]

    """ start to train the model """
    # start
    startTime = time.time()

    model = Sequential()
    
    # add RBF layer
    rbflayer = RBFLayer(noNeurons,
                        initializer=InitCentersRandom(inputSamples),
                        betas=50.0,
                        input_shape=(inputNo,))

#    os.system("pause")
    model.add(rbflayer)

    """    
    rbflayer2 = RBFLayer(noNeurons,
                         initializer=None,
                         betas=50.0,
                         input_shape=(noNeurons,))
    model.add(rbflayer2)
    """

#    model.add(Dense(noNeurons, input_dim=noNeurons, kernel_initializer="normal",activation="relu"))
    
    # add the output layer
    denselayer = Dense(outputNo, input_dim=noNeurons, kernel_initializer="normal",activation="linear")
    model.add(denselayer)
    
#    centers_ori = rbflayer.get_weights()[0]

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=50, min_lr=1.0e-8)

    e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=100, verbose=1, mode='auto')

    filepath="./model/model_sf_{epoch:02d}_{loss:.6f}_{val_loss:.6f}.hdf5"

    chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=500)

    # Compile model
    opt = Adam(lr=0.001, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt)


    hist = model.fit(inputSamples, outputSamples, batch_size=64,validation_split=0.1,\
                     callbacks=[reduce_lr,e_stop,chkpt],epochs=1000, verbose=1)

    #save model
    model.save('./model/final_sf.hdf5') 
    
    print("\n")
    print("loss = %e to %e"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))

    print("--------------------------------")
    print("Training duration:\n", time.time()-startTime)
    print("--------------------------------")
    
    # save the hyper parameters of the neural network
    centers = rbflayer.get_weights()[0]
    widths = rbflayer.get_weights()[1]
    weights = denselayer.get_weights()[0]
    biases = denselayer.get_weights()[1]
    
    parameters = [centers,widths,weights,biases,iMin,iMax,oMin,oMax,np.asarray(hist.history["loss"])]
    with open("hy"+str(noNeurons)+"neurons.pkl","wb") as outfile:
        pickle.dump(parameters, outfile, pickle.HIGHEST_PROTOCOL)
        
        
    data1=[hist.history]
    with open('./model/hist.pkl', 'wb') as outfile:
        pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)    


