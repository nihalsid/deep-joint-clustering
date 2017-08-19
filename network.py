'''
Created on Jul 11, 2017

@author: yawarnihal, eliealjalbout
'''

from datetime import datetime
import logging

from lasagne import layers
import lasagne
from lasagne.layers.helper import get_all_layers
import theano

from customlayers import ClusteringLayer, Unpool2DLayer
from misc import evaluateKMeans, visualizeData, rescaleReshapeAndSaveImage
import numpy as np
import theano.tensor as T

from lasagne.layers import batch_norm

logFormatter = logging.Formatter("[%(asctime)s]  %(message)s", datefmt='%m/%d %I:%M:%S')

rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(datetime.now().strftime('dcjc_%H_%M_%d_%m.log'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


class DCJC(object):
    def __init__(self, network_description):

        self.name = network_description['name']
        netbuilder = NetworkBuilder(network_description)
        self.network = netbuilder.buildNetwork()
        self.encode_layer, self.encode_size = netbuilder.getEncodeLayerAndSize()
        self.t_input, self.t_target = netbuilder.getInputAndTargetVars()
        self.input_type = netbuilder.getInputType()
        rootLogger.info("Network: " + self.networkToStr())
        recon_prediction_expression = layers.get_output(self.network)
        encode_prediction_expression = layers.get_output(self.encode_layer, deterministic=True)
        loss = self.getReconstructionLossExpression(recon_prediction_expression, self.t_target)
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(loss, params)
        self.trainAutoencoder = theano.function([self.t_input, self.t_target], loss, updates=updates)
        self.predictReconstruction = theano.function([self.t_input], recon_prediction_expression)
        self.predictEncoding = theano.function([self.t_input], encode_prediction_expression)

    def pretrainWithData(self, dataset, pretrain_epochs):
        batch_size = 60
        Z = np.zeros((dataset.input.shape[0], self.encode_size), dtype=np.float32);
        for epoch in range(pretrain_epochs):
            pretrain_error = 0
            pretrain_total_batches = 0
            for batch in dataset.iterate_minibatches(self.input_type, batch_size, shuffle=True):
                inputs, targets = batch
                pretrain_error += self.trainAutoencoder(inputs, targets)
                pretrain_total_batches += 1
            if (epoch + 1) % 5 == 0:
                for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                    Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding(batch[0])
                    '''
                    for i, x in enumerate(self.predictReconstruction(batch[0])):
                        rescaleReshapeAndSaveImage(x[0], "dumps/%02d%03d.jpg"%(epoch,i));
                    '''
                rootLogger.info(evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), "%d/%d [%.4f]" % (
                    epoch + 1, pretrain_epochs, pretrain_error / pretrain_total_batches))[0])
            else:
                rootLogger.info("%-30s     %8s     %8s" % (
                    "%d/%d [%.4f]" % (epoch + 1, pretrain_epochs, pretrain_error / pretrain_total_batches), "", ""))

        for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
            Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding(batch[0])

        np.save('saved_params/%s/z_%s.npy' % (dataset.name, self.name), Z)
        np.savez('saved_params/%s/m_%s.npz' % (dataset.name, self.name),
                 *lasagne.layers.get_all_param_values(self.network))

    # load pretrained models, then either train with DEC loss jointly with reconstruction or alone
    def doClustering(self, dataset, complete_loss, cluster_train_epochs, repeats):
        P = T.matrix('P')
        batch_size = 60
        with np.load('saved_params/%s/m_%s.npz' % (dataset.name, self.name)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)
        Z = np.load('saved_params/%s/z_%s.npy' % (dataset.name, self.name))
        quality_desc, cluster_centers = evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), 'Initial')
        rootLogger.info(quality_desc)
        dec_network = ClusteringLayer(self.encode_layer, dataset.getClusterCount(), cluster_centers, batch_size,
                                      self.encode_size)
        dec_output_exp = layers.get_output(dec_network, deterministic=True)
        encode_output_exp = layers.get_output(self.network, deterministic=True)
        clustering_loss = self.getClusteringLossExpression(dec_output_exp, P)
        reconstruction_loss = self.getReconstructionLossExpression(encode_output_exp, self.t_target)
        params_ae = lasagne.layers.get_all_params(self.network, trainable=True)
        params_dec = lasagne.layers.get_all_params(dec_network, trainable=True)

        w_cluster_loss = 1
        w_reconstruction_loss = 1
        total_loss = w_cluster_loss * clustering_loss
        if (complete_loss):
            total_loss = total_loss + w_reconstruction_loss * reconstruction_loss
        all_params = params_dec
        if complete_loss:
            all_params.extend(params_ae)
        all_params = list(set(all_params))

        updates = lasagne.updates.adam(total_loss, all_params)

        getSoftAssignments = theano.function([self.t_input], dec_output_exp)

        trainFunction = None
        if complete_loss:
            trainFunction = theano.function([self.t_input, self.t_target, P], total_loss, updates=updates)
        else:
            trainFunction = theano.function([self.t_input, P], clustering_loss, updates=updates)

        for _iter in range(repeats):
            qij = np.zeros((dataset.input.shape[0], dataset.getClusterCount()), dtype=np.float32)
            for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                qij[i * batch_size: (i + 1) * batch_size] = getSoftAssignments(batch[0])
            # np.save('saved_params/%s/q_%s.npy' % (dataset.name, 'Test'), qij)
            pij = self.calculateP(qij)
            # np.save('saved_params/%s/p_%s.npy' % (dataset.name, 'Test'), pij)

            for _epoch in range(cluster_train_epochs):
                cluster_train_error = 0
                cluster_train_total_batches = 0
                for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, pij, shuffle=True)):
                    if (complete_loss):
                        cluster_train_error += trainFunction(batch[0], batch[0], batch[1])
                    else:
                        cluster_train_error += trainFunction(batch[0], batch[1])
                    cluster_train_total_batches += 1
                    #### CAN REMOVE - MORE THAN 1 EPOCHS WERE GIVING BAD RESULTS ####
                '''
                for i, batch in enumerate(dataset.iterate_minibatches(train_set, batch_size, shuffle=False)):
                    Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding(batch[0])
                rootLogger.info(evaluateKMeans(Z, dataset.labels, "%d.%d/%d.%d [%.4f]" % (_iter, _epoch + 1, repeats, cluster_train_epochs, cluster_train_error / cluster_train_total_batches))[0])
                '''
                #################################################################
            for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding(batch[0])

            # np.save('saved_params/%s/z_%s.npy' % (dataset.name, 'Test'), Z)
            rootLogger.info(evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), "%d/%d [%.4f]" % (
                _iter, repeats, cluster_train_error / cluster_train_total_batches))[0])

    def calculateP(self, Q):
        f = Q.sum(axis=0)
        pij_numerator = Q * Q
        pij_numerator = pij_numerator / f
        normalizer_p = pij_numerator.sum(axis=1).reshape((Q.shape[0], 1))
        P = pij_numerator / normalizer_p
        return P

    def getClusteringLossExpression(self, Q_expression, P_expression):
        log_arg = P_expression / Q_expression
        log_exp = T.log(log_arg)
        sum_arg = P_expression * log_exp
        loss = sum_arg.sum(axis=1).sum(axis=0)
        return loss

    def getReconstructionLossExpression(self, prediction_expression, t_target):
        loss = lasagne.objectives.squared_error(prediction_expression, t_target)
        loss = loss.mean()
        return loss

    def networkToStr(self):
        layers = lasagne.layers.get_all_layers(self.network)
        result = ''
        for layer in layers:
            t = type(layer)
            if t is lasagne.layers.input.InputLayer:
                pass
            else:
                result += ' ' + layer.name
        return result.strip()


class NetworkBuilder(object):
    def __init__(self, network_description):
        self.network_description = self.populateMissingDescriptions(network_description)
        if self.network_description['network_type'] == 'CAE':
            self.t_input = T.tensor4('input_var')
            self.t_target = T.tensor4('target_var')
            self.input_type = "IMAGE"
        else:
            self.t_input = T.matrix('input_var')
            self.t_target = T.matrix('target_var')
            self.input_type = "FLAT"
        self.network_type = self.network_description['network_type']
        self.batch_norm = bool(self.network_description["use_batch_norm"])

    def getInputAndTargetVars(self):
        return self.t_input, self.t_target

    def getInputType(self):
        return self.input_type

    def buildNetwork(self):
        network = None
        for layer in self.network_description['layers']:
            network = self.processLayer(network, layer)
        return network

    def getEncodeLayerAndSize(self):
        return self.encode_layer, self.encode_size

    def populateDecoder(self, encode_layers):
        decode_layers = []
        for i, layer in reversed(list(enumerate(encode_layers))):
            if (layer["type"] == "MaxPool2D*"):
                decode_layers.append({
                    "type": "InverseMaxPool2D",
                    "layer_index": i,
                    'filter_size': layer['filter_size']
                })
            elif (layer["type"] == "MaxPool2D"):
                decode_layers.append({
                    "type": "Unpool2D",
                    'filter_size': layer['filter_size']
                })
            elif (layer["type"] == "Conv2D"):
                decode_layers.append({
                    'type': 'Deconv2D',
                    'non_linearity': layer['non_linearity'],
                    'filter_size': layer['filter_size'],
                    'num_filters': encode_layers[i - 1]['output_shape'][0]
                })
            elif (layer["type"] == "Dense"):
                decode_layers.append({
                    'type': 'Dense',
                    'num_units': encode_layers[i - 1]['output_shape'][2]
                })
        encode_layers.extend(decode_layers)

    def populateShapes(self, layers):
        last_layer_dimensions = layers[0]['output_shape']
        for layer in layers[1:]:
            if (layer['type'] == 'MaxPool2D'):
                layer['output_shape'] = [last_layer_dimensions[0], last_layer_dimensions[1] / layer['filter_size'][0],
                                         last_layer_dimensions[2] / layer['filter_size'][1]]
            elif (layer['type'] == 'Conv2D'):
                layer['output_shape'] = [layer['num_filters'],
                                         last_layer_dimensions[1] - (layer['filter_size'][0] - 1) * 1,
                                         last_layer_dimensions[2] - (layer['filter_size'][1] - 1) * 1]
            elif (layer['type'] == 'Dense'):
                layer['output_shape'] = [1, 1, layer['num_units']]
            last_layer_dimensions = layer['output_shape']

    def populateMissingDescriptions(self, network_description):
        network_description['layers'][-1]['is_encode'] = True
        self.populateShapes(network_description['layers'])
        self.populateDecoder(network_description['layers'])
        if 'use_batch_norm' not in network_description:
            network_description['use_batch_norm'] = False
        if 'network_type' not in network_description:
            if (network_description['name'].split('_')[0].split('-')[0] == 'fc'):
                network_description['network_type'] = 'AE'
            else:
                network_description['network_type'] = 'CAE'
        for layer in network_description['layers']:
            if 'conv_mode' not in layer:
                layer['conv_mode'] = 'valid'
            if 'is_encode' not in layer:
                layer['is_encode'] = False
        return network_description

    def processLayer(self, network, layer_definition):
        if (layer_definition["type"] == "Input"):
            if self.network_type == 'CAE':
                network = lasagne.layers.InputLayer(
                    shape=tuple([None] + layer_definition['output_shape']), input_var=self.t_input)
            elif self.network_type == 'AE':
                network = lasagne.layers.InputLayer(
                    shape=(None, layer_definition['output_shape'][2]), input_var=self.t_input)
        elif (layer_definition['type'] == 'Dense'):
            lasagne.layers.DenseLayer(network, num_units=layer_definition['num_units'],
                                      nonlinearity=self.getNonLinearity(layer_definition['non_linearity']),
                                      name=self.getLayerName(layer_definition))
        elif (layer_definition['type'] == 'Conv2D'):
            network = lasagne.layers.Conv2DLayer(network, num_filters=layer_definition['num_filters'],
                                                 filter_size=tuple(layer_definition["filter_size"]),
                                                 pad=layer_definition['conv_mode'],
                                                 nonlinearity=self.getNonLinearity(layer_definition['non_linearity']),
                                                 name=self.getLayerName(layer_definition))
        elif (layer_definition['type'] == 'MaxPool2D'):
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=tuple(layer_definition["filter_size"]),
                                                    name=self.getLayerName(layer_definition))
        elif (layer_definition['type'] == 'InverseMaxPool2D'):
            network = lasagne.layers.InverseLayer(network, get_all_layers(network)[layer_definition['layer_index']],
                                                  name=self.getLayerName(layer_definition))
        elif (layer_definition['type'] == 'Unpool2D'):
            network = Unpool2DLayer(network, tuple(layer_definition['filter_size']),
                                    name=self.getLayerName(layer_definition))
        elif (layer_definition['type'] == 'Deconv2D'):
            network = lasagne.layers.Deconv2DLayer(network, num_filters=layer_definition['num_filters'],
                                                   filter_size=tuple(layer_definition['filter_size']),
                                                   crop=layer_definition['conv_mode'],
                                                   nonlinearity=self.getNonLinearity(layer_definition['non_linearity']),
                                                   name=self.getLayerName(layer_definition))
        if (self.batch_norm and layer_definition['type'] in ("Conv2D", "Deconv2D", "Dense")):
            network = batch_norm(network)

        if (layer_definition['is_encode']):
            self.encode_layer = lasagne.layers.flatten(network, name='fl')
            self.encode_size = layer_definition['output_shape'][0] * layer_definition['output_shape'][1] * \
                               layer_definition['output_shape'][2]
        return network

    def getLayerName(self, layer_definition):
        if (layer_definition['type'] == 'Dense'):
            return 'fc[{}]'.format(layer_definition['num_units'])
        elif (layer_definition['type'] == 'Conv2D'):
            return '{}[{}]'.format(layer_definition['num_filters'],
                                   'x'.join([str(fs) for fs in layer_definition['filter_size']]))
        elif (layer_definition['type'] == 'MaxPool2D'):
            return 'max[{}]'.format('x'.join([str(fs) for fs in layer_definition['filter_size']]))
        elif (layer_definition['type'] == 'InverseMaxPool2D'):
            return 'ups*[{}]'.format('x'.join([str(fs) for fs in layer_definition['filter_size']]))
        elif (layer_definition['type'] == 'Unpool2D'):
            return 'ups[{}]'.format(
                str(layer_definition['filter_size'][0]) + 'x' + str(layer_definition['filter_size'][1]))
        elif (layer_definition['type'] == 'Deconv2D'):
            return '{}[{}]'.format(layer_definition['num_filters'],
                                   'x'.join([str(fs) for fs in layer_definition['filter_size']]))

    def getNonLinearity(self, non_linearity):
        return {
            'rectify': lasagne.nonlinearities.rectify,
            'linear': lasagne.nonlinearities.linear,
            'elu': lasagne.nonlinearities.elu
        }[non_linearity]
