'''
Created on Jul 9, 2017

@author: yawarnihal
'''
import cPickle
import gzip
import time

from lasagne import layers
import lasagne
from nolearn.lasagne import BatchIterator
import theano

import numpy as np
import shape
import theano.tensor as T

def refineDatasetForAutoencoder(dataset):
    X = dataset
    X = np.rint(X * 256).astype(np.int).reshape((-1, 1, 28, 28))  # convert to (0,255) int range (we'll do our own scaling)
    mu, sigma = np.mean(X.flatten()), np.std(X.flatten())
    X_train = X.astype(np.float64)
    X_train = (X_train - mu) / sigma
    X_train = X_train.astype(np.float32)
    return (X_train, X_train)

def loadDataSet(fname='mnist/mnist.pkl.gz'):
    f = gzip.open(fname, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    (X_train_in, X_train_out) = refineDatasetForAutoencoder(train_set[0])
    (X_valid_in, X_valid_out) = refineDatasetForAutoencoder(valid_set[0])
    return (X_train_in, X_train_out, X_valid_in, X_valid_out)

class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):
        super(Unpool2DLayer, self).__init__(incoming, **kwargs)
        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds
    
    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]
        return tuple(output_shape)

    def get_output_for(self, incoming, **kwargs):
        ds = self.ds
        return incoming.repeat(ds[0], axis=2).repeat(ds[1], axis=3)
    

class FlipBatchIterator(BatchIterator):

    def transform(self, X1, X2):
        X1b, X2b = super(FlipBatchIterator, self).transform(X1, X2)
        X2b = X2b.reshape(X1b.shape)

        bs = X1b.shape[0]
        h_indices = np.random.choice(bs, bs / 2, replace=False)  # horizontal flip
        v_indices = np.random.choice(bs, bs / 2, replace=False)  # vertical flip

        ###  uncomment these lines if you want to include rotations (images must be square)  ###
        # r_indices = np.random.choice(bs, bs / 2, replace=False) # 90 degree rotation
        for X in (X1b, X2b):
            X[h_indices] = X[h_indices, :, :, ::-1]
            X[v_indices] = X[v_indices, :, ::-1, :]
            # X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3)
        shape = X2b.shape
        X2b = X2b.reshape((shape[0], -1))

        return X1b, X2b

def CreateDCJC(input_dim, filter_defs, input_var=None):
    # Filter defs 
    # List of filters ((numberoffilters, filtershape_x, filtershape_y),...)
    # Make it more general to loop over instead of init 1 by 1
    network = lasagne.layers.InputLayer(shape=(None, 1, input_dim[0], input_dim[1]), input_var=input_var)
    shape = network.get_output_shape_for((500, 1, input_dim[0], input_dim[1]))
    print shape
    network = lasagne.layers.Conv2DLayer(network, num_filters=filter_defs[0][0], filter_size=(filter_defs[0][1], filter_defs[0][2]), nonlinearity=lasagne.nonlinearities.tanh, W=lasagne.init.GlorotUniform())
    shape = network.get_output_shape_for(shape)
    print shape
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(filter_defs[1][0], filter_defs[1][1]))
    shape = network.get_output_shape_for(shape)
    print shape
    # network = lasagne.layers.DenseLayer(network, num_units=(filter_defs[0][0]*input_dim[0]*input_dim[1])/4, nonlinearity=lasagne.nonlinearities.tanh)
    network = Unpool2DLayer(network, (filter_defs[1][0], filter_defs[1][0]))
    shape = network.get_output_shape_for(shape)
    print shape
    network = lasagne.layers.Deconv2DLayer(network, num_filters=1, filter_size=(filter_defs[0][1], filter_defs[0][2]), nonlinearity=lasagne.nonlinearities.tanh, W=lasagne.init.GlorotUniform())
    shape = network.get_output_shape_for(shape)
    print shape
    return network

input_var = T.tensor4('input')
target_var = T.tensor4('target')

network = CreateDCJC((28, 28), [[32, 7, 7], [2, 2]], input_var)
prediction = lasagne.layers.get_output(network) 
loss = lasagne.objectives.squared_error(prediction, target_var)
loss = loss.mean()
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.squared_error(prediction, target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var))
train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

print("Starting training...")
# We iterate over epochs:

dataset = loadDataSet()
print dataset[1].shape
num_epochs = 5
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    start_time = time.time()
    train_err = train_fn(dataset[0], dataset[1])

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_err, val_acc = val_fn(dataset[2], dataset[3])
    
    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err))
    print("  validation loss:\t\t{:.6f}".format(val_err))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc * 100))
