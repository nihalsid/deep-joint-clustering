'''
Created on Jul 11, 2017

@author: yawarnihal
'''
from lasagne import layers
import lasagne
from lasagne.layers.helper import get_all_layers
from sklearn import metrics
from sklearn.cluster.k_means_ import KMeans
import theano

import numpy as np
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
    
invertible_layers = [layers.Conv2DLayer, layers.MaxPool2DLayer]

class DCJC(object):
    
    def __init__(self, network_description):
        self.input_var = T.tensor4('input_var')
        self.target_var = T.tensor4('target_var')
        self.network = self.getNetworkExpression(network_description)
        recon_prediction_expression = self.getReconstructionPredictionExpression(self.network)
        encode_prediction_expression = self.getEncodePredictionExpression(self.encode_layer)
        loss = self.getLossExpression(recon_prediction_expression, self.target_var)
        updates = self.getNetworkUpdates(self.network, loss)
        self.train = self.getTrainFunction(self.input_var, self.target_var, loss, updates)
        self.predictReconstruction = self.getReconstructionPredictionFunction(self.input_var, recon_prediction_expression)
        self.predictEncoding = self.getEncodePredictionFunction(self.input_var, encode_prediction_expression)
        self.validate = self.getValidationFunction(self.input_var, self.target_var, loss)
        
    def getNonLinearity(self, non_linearity_name):
        return {
                'rectify': lasagne.nonlinearities.rectify,
                }[non_linearity_name]
    
    def getLayer(self, network, layer_definition, is_encode_layer):
        if (layer_definition['layer_type'] == 'Conv2D'):
            network = layers.Conv2DLayer(network, num_filters=layer_definition['num_filters'], filter_size=(layer_definition['filter_size'][0], layer_definition['filter_size'][1]), nonlinearity=self.getNonLinearity(layer_definition['non_linearity']), W=lasagne.init.GlorotUniform())
            if (is_encode_layer):
                self.encode_layer = lasagne.layers.flatten(network)
                self.encode_size = layer_definition['output_shape'][0] * layer_definition['output_shape'][1] * layer_definition['output_shape'][2]
            return network
        elif (layer_definition['layer_type'] == 'MaxPool2D'):
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(layer_definition['filter_size'][0], layer_definition['filter_size'][1]))
            if (is_encode_layer):
                self.encode_layer = lasagne.layers.flatten(network)
                self.encode_size = layer_definition['output_shape'][0] * layer_definition['output_shape'][1] * layer_definition['output_shape'][2]
            return network
        elif (layer_definition['layer_type'] == 'Encode'):
            network = lasagne.layers.flatten(network)
            network = lasagne.layers.DenseLayer(network, num_units=layer_definition['encode_size'], nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.GlorotUniform())
            self.encode_layer = network
            self.encode_size = layer_definition['encode_size']
            network = lasagne.layers.DenseLayer(network, num_units=layer_definition['output_shape'][0] * layer_definition['output_shape'][1] * layer_definition['output_shape'][2], nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
            return lasagne.layers.reshape(network, (-1, layer_definition['output_shape'][0], layer_definition['output_shape'][1], layer_definition['output_shape'][2]))
        elif (layer_definition['layer_type'] == 'Unpool2D'):
            return Unpool2DLayer(network, (layer_definition['filter_size'][0], layer_definition['filter_size'][1]))
        elif (layer_definition['layer_type'] == 'Deconv2D'):
            return layers.Deconv2DLayer(network, num_filters=layer_definition['num_filters'], filter_size=(layer_definition['filter_size'][0], layer_definition['filter_size'][1]), nonlinearity=self.getNonLinearity(layer_definition['non_linearity']), W=lasagne.init.GlorotUniform())
        elif (layer_definition['layer_type'] == 'Input'):
            return layers.InputLayer(shape=(None, layer_definition['output_shape'][0], layer_definition['output_shape'][1], layer_definition['output_shape'][2]), input_var=self.input_var)
    
    def populateNetworkOutputShapes(self, network_description):
        last_layer_dimensions = network_description['layers_encode'][0]['output_shape']
        for layer in network_description['layers_encode']:
            if (layer['layer_type'] == 'MaxPool2D'):
                layer['output_shape'] = [last_layer_dimensions[0], last_layer_dimensions[1] / layer['filter_size'][0], last_layer_dimensions[2] / layer['filter_size'][1]]
            elif (layer['layer_type'] == 'Conv2D'):
                layer['output_shape'] = [layer['num_filters'], last_layer_dimensions[1] - layer['filter_size'][0] + 1 , last_layer_dimensions[2] - layer['filter_size'][1] + 1]
            elif (layer['layer_type'] == 'Encode'):
                layer['output_shape'] = last_layer_dimensions 
            last_layer_dimensions = layer['output_shape']

    def populateMirroredNetwork(self, network_description):
        network_description['layers_decode'] = [] 
        old_network_description = network_description.copy()
        for i in range(len(old_network_description['layers_encode']) - 1, -1, -1):
            if (old_network_description['layers_encode'][i]['layer_type'] == 'MaxPool2D'):
                old_network_description['layers_decode'].append({
                                                             'layer_type':'Unpool2D',
                                                             'filter_size':old_network_description['layers_encode'][i]['filter_size']
                                                             })
            elif(old_network_description['layers_encode'][i]['layer_type'] == 'Conv2D'):
                network_description['layers_decode'].append({
                                                             'layer_type':'Deconv2D',
                                                             'non_linearity': old_network_description['layers_encode'][i]['non_linearity'],
                                                             'filter_size':old_network_description['layers_encode'][i]['filter_size'],
                                                             'num_filters':old_network_description['layers_encode'][i - 1]['output_shape'][0]
                                                             })
        
    def getNetworkExpression(self, network_description):
        network = None
        self.populateNetworkOutputShapes(network_description)
        for i, layer in enumerate(network_description['layers_encode']):
            network = self.getLayer(network, layer, i==len(network_description['layers_encode'])-1)
            # print network
        layer_list = get_all_layers(network)
        if network_description['use_inverse_layers'] == True:
            for i in range(len(layer_list) - 1, 0, -1):
                # print network, layer_list[i]
                if any(type(layer_list[i]) is invertible_layer for invertible_layer in invertible_layers):
                    network = lasagne.layers.InverseLayer(network, layer_list[i])
        else:
            self.populateMirroredNetwork(network_description)
            for layer in network_description['layers_decode']:
                network = self.getLayer(network, layer)
                # print network
        return network
    
    def getReconstructionPredictionExpression(self, network):
        return layers.get_output(network)
    
    def getEncodePredictionExpression(self, encode_layer):
        return layers.get_output(encode_layer)
    
    def getLossExpression(self, prediction_expression, target_var):
        loss = lasagne.objectives.squared_error(prediction_expression, target_var)
        loss = loss.mean()
        return loss
        
    def getNetworkUpdates(self, network, loss):
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
        return updates
    
    def getTrainFunction(self, input_var, output_var, loss, updates):
        return theano.function([input_var, output_var], loss, updates=updates)
    
    def getReconstructionPredictionFunction(self, input_var, prediction_expression):
        return theano.function([input_var], prediction_expression)
    
    def getEncodePredictionFunction(self, input_var, encode_expression):
        return theano.function([input_var], encode_expression)
    
    def getValidationFunction(self, input_var, output_var, loss):
        return theano.function([input_var, output_var], loss)
    
    def pretrainWithData(self, dataset, arch_name, pretrain_epochs):
        pretrain_error = 0
        pretrain_total_batches = 0
        batch_size = 250
        for epoch in range(pretrain_epochs):
            for batch in dataset.iterate_minibatches('train', batch_size, shuffle=True):
                inputs, targets = batch
                pretrain_error += self.train(inputs, targets)
                pretrain_total_batches += 1
            print 'Done with epoch %d/%d [TE: %.4f]' % (epoch + 1, pretrain_epochs, pretrain_error / pretrain_total_batches)
        
        Z = np.zeros((dataset.train_input.shape[0], self.encode_size), dtype=np.float32);
        print Z.shape
        Z = self.predictEncoding(dataset.train_input)
        Z.dump('models/z_%s.pkl'%arch_name)
        np.savez('models/m_%s.npz'%arch_name, *lasagne.layers.get_all_param_values(self.network))
        # Assume training complete
        # After training is complete get encodings for all the train data
        # Next do kmeans to find centers of clusters
        # With cluster centers, create the new layer - the clustering layer with input as the 
        # z layer and output as the q distribution, with weights = cluster centers
        # Loss for this new layer = kl divergence and reconstruction loss
    
    def doClustering(self, dataset):
        with np.load('model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)
        Z = np.load('z_dump.pkl')
        kmeans = KMeans(init='k-means++', n_clusters=10)
        kmeans.fit(Z)
        print '%8.3f\t %8.3f\t %8.3f\t %8.3f\t %8.3f' % (metrics.homogeneity_score(dataset.train_labels, kmeans.labels_),
             metrics.completeness_score(dataset.train_labels, kmeans.labels_),
             metrics.v_measure_score(dataset.train_labels, kmeans.labels_),
             metrics.adjusted_rand_score(dataset.train_labels, kmeans.labels_),
             metrics.adjusted_mutual_info_score(dataset.train_labels, kmeans.labels_))
    
    def printLayers(self):
        layers = get_all_layers(self.network)
        # shape_i = (500,1,28,28)
        for l in layers:
            print type(l)
            # print l.get_output_shape_for(shape_i)
            # shape_i = l.get_output_shape_for(shape_i)
