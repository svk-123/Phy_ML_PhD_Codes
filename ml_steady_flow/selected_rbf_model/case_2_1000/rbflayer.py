# -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Orthogonal, Constant
import numpy as np

class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.

    # Arguments
        X: matrix, dataset to choose the centers from (random rows 
          are taken as centers)
    """
    def __init__(self, X):
        self.X = X 

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        # randomly select the initial values of centers
        idx = np.random.randint(self.X.shape[0], size=shape[0])

        # uniformly select the initial values of centers
#        idx = np.floor(np.mgrid[0:(self.X.shape[0]-1):complex(0, shape[0])])
#        idx = np.array([np.int(ids) for ids in idx])

        return self.X[idx,:]

class InitCentersKeras(Initializer):
    """ Initializer for initialization of centers of the first hidden layer in
        multi-layer RBF network using k-means clustering from the given data set.

    # Arguments
        X: matrix, dataset to choose the centers from
    """
    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        """ k-means """
        k = shape[0]
        
        clusters = self.X[np.random.randint(self.X.shape[0], size=k), :]

        prevClusters = clusters.copy()
        stds = np.zeros(k)
        converged = False
        distances = np.zeros((self.X.shape[0], k))

        while not converged:
            """
            compute distances for each cluster center to each point 
            where (distances[i, j] represents the distance between the ith point and jth cluster)
            """
            for i in range(self.X.shape[0]):
                for j in range(k):
                    distances[i,j] = np.linalg.norm(self.X[i,:] - clusters[j,:])
            
            # find the cluster that's closest to each point
            closestCluster = np.argmin(distances, axis=1)

            # update clusters by taking the mean of all of the points assigned to that cluster
            for i in range(k):
                pointsForCluster = self.X[closestCluster == i]
                if len(pointsForCluster) > 0:
                    clusters[i] = np.mean(pointsForCluster, axis=0)

            # converge if clusters haven't moved
            converged = np.linalg.norm(clusters - prevClusters) < 1e-6
            prevClusters = clusters.copy()
            
            # end of while not converged

        for i in range(self.X.shape[0]):
            for j in range(k):
                distances[i,j] = np.linalg.norm(self.X[i,:] - clusters[j,:])

        closestCluster = np.argmin(distances, axis=1)

        clustersWithNoPoints = []
        for i in range(k):
            pointsForCluster = self.X[closestCluster == i]
            if len(pointsForCluster) < 2:
                # keep track of clusters with no points or 1 point
                clustersWithNoPoints.append(i)
                continue
            else:
                stds[i] = np.std(self.X[closestCluster == i])

        # if there are clusters with 0 or 1 points, take the mean std of the other clusters
        if len(clustersWithNoPoints) > 0:
            pointsToAverage = []
            for i in range(k):
                if i not in clustersWithNoPoints:
                    pointsToAverage.append(self.X[closestCluster == i])
            pointsToAverage = np.concatenate(pointsToAverage).ravel()
            stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

        return clusters.reshape(k, self.X.shape[1])
    

        
class RBFLayer(Layer):
    """ Layer of Gaussian RBF units. 

    # Example
 
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X), 
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    

    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas 

    """
    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
            #self.initializer = Orthogonal()
        else:
            self.initializer = initializer 
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.centers = self.add_weight(name='centers', 
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(value=self.init_betas),
                                     #initializer='ones',
                                     trainable=True)
            
        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp( -self.betas * K.sum(H**2, axis=1))
        
        #C = self.centers[np.newaxis, :, :]
        #X = x[:, np.newaxis, :]

        #diffnorm = K.sum((C-X)**2, axis=-1)
        #ret = K.exp( - self.betas * diffnorm)
        #return ret 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
