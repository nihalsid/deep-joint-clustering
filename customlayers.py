'''
Created on Jul 25, 2017

@author: eliealjalbout,yawarnihal
'''

from lasagne import layers
import theano
import theano.tensor as T

class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):
        
        super(Unpool2DLayer, self).__init__(incoming, **kwargs)
        self.ds = ds
    
    def get_output_shape_for(self, input_shape):
        
        output_shape = list(input_shape)
        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]
        return tuple(output_shape)

    def get_output_for(self, incoming, **kwargs):
        
        ds = self.ds
        return incoming.repeat(ds[0], axis=2).repeat(ds[1], axis=3)

class ClusteringLayer(layers.Layer):

    def __init__(self, incoming, num_clusters, initial_clusters, num_samples, latent_space_dim, **kwargs):
        
        super(ClusteringLayer, self).__init__(incoming, **kwargs)
        self.num_clusters = num_clusters
        self.W = self.add_param(theano.shared(initial_clusters), initial_clusters.shape, 'W')
        self.num_samples = num_samples
        self.latent_space_dim = latent_space_dim
        
    def get_output_shape_for(self, input_shape):
        
        return (input_shape[0], self.num_clusters)
    
    """
    params:
    -z: corresponds to cluster assignement 
    -u: corresponds to cluster center 
    """
    def get_output_for(self, incoming, **kwargs):

        z_expanded = incoming.reshape((self.num_samples, 1, self.latent_space_dim))
        z_expanded = T.tile(z_expanded, (1, self.num_clusters, 1)) 
        u_expanded = T.tile(self.W, (self.num_samples, 1, 1))
        
        distances_from_cluster_centers = (z_expanded - u_expanded).norm(2, axis=2)
        qij_numerator = 1 + distances_from_cluster_centers * distances_from_cluster_centers
        qij_numerator = 1 / qij_numerator
        normalizer_q = qij_numerator.sum(axis=1).reshape((self.num_samples, 1))
        
        return qij_numerator / normalizer_q;