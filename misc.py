'''
Created on Jul 11, 2017

@author: yawarnihal, eliealjalbout
'''

import cPickle
import gzip

from PIL import Image

import numpy as np


class Dataset(object):
    
    def loadDataSet(self, fname='mnist/mnist.pkl.gz'):
        f = gzip.open(fname, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        self.train_input, self.train_target, self.train_mu, self.train_sigma = self.prepareDatasetForAutoencoder(train_set[0])
        self.valid_input, self.valid_target, self.valid_mu, self.valid_sigma = self.prepareDatasetForAutoencoder(valid_set[0])
        self.test_input, self.test_target, self.test_mu, self.test_sigma = self.prepareDatasetForAutoencoder(test_set[0])
    
    def prepareDatasetForAutoencoder(self, dataset):
        X = dataset
        X = np.rint(X * 256).astype(np.int).reshape((-1, 1, 28, 28))
        mu, sigma = np.mean(X.flatten()), np.std(X.flatten())
        preprocessed_X = X.astype(np.float64)
        preprocessed_X = (preprocessed_X - mu) / sigma
        preprocessed_X = preprocessed_X.astype(np.float32)
        return (preprocessed_X, preprocessed_X, mu, sigma)

    def iterate_minibatches(self, set_type, batch_size, shuffle=False):
        inputs = None
        targets = None
        if set_type == 'train':
            inputs = self.train_input
            targets = self.train_target
        elif set_type == 'validation':
            inputs = self.valid_input
            targets = self.valid_target
        elif set_type == 'test':
            inputs = self.test_input
            targets = self.test_target
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]

def rescaleReshapeAndSaveImage(image_sample, mu, sigma, out_filename):
    #print image_sample.shape
    #print mu.shape
    #print sigma.shape
    image_sample = image_sample * sigma + mu;
    image_sample = np.rint(image_sample).astype(int)
    image_sample = np.clip(image_sample, a_min=0, a_max=255).astype('uint8')
    img = Image.fromarray(image_sample, 'L')
    img.save(out_filename)
