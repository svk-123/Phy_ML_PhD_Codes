# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import Adam
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm
import time
import pickle
#import os


def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * np.linalg.norm(x-c)**2)

""" function: load_data """
def load_data(fileDir, ReNo, fileExt):
    
    fileFullDir = fileDir + str(ReNo) + fileExt

    with open(fileFullDir, "rb") as fid:
        data = pickle.load(fid)

    return data
""" function: load_data """



""" main function """
if __name__ == "__main__":

    # load flow field datasets from data directory
    # data: x, y, ReNo, u, v, p
    
    ReNos = np.array([8000])
    inputNo = 3
    outputNo = 3
    noNeurons = 4000
    
    inputSamples = np.zeros((1,inputNo))
    outputSamples = np.zeros((1,outputNo))

    for ii in range(len(ReNos)):
        data = load_data("../../data/cavity_Re", ReNos[ii], ".pkl")
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
        
            
        with open('hy4000neurons.pkl', "rb") as fid:
            para=pickle.load(fid)
            iMin = para[4]
            iMax = para[5]
            oMin = para[6]
            oMax = para[7]
        iDiff = iMax - iMin
        oDiff = oMax - oMin
           
        
        for i in range(sampleNo):
            for j in range(inputNo):
                inputSamples[i, j] = (inputSamples[i, j] -iMin[j])/iDiff[j]
                
        # normalize - output
        oriOutput = outputSamples.copy()
        
        """ plot original input flow field """
    #    plot(oriInput[:,0],oriInput[:,1],oriOutput[:,0],20,'Re100/Re100-u-CFD')
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
        
        model.load_weights('./model/final_sf.hdf5')
        
        outputPredicted = model.predict(inputSamples)
        for i in range(sampleNo):
            for j in range(outputNo):
                outputPredicted[i,j] = outputPredicted[i,j]*oDiff[j] +oMin[j]
    

    
        data1=[outputPredicted]
        with open('./pred/Re%s.pkl'%ReNos[ii], 'wb') as outfile:
            pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)   


