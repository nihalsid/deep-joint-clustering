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
        self.train_input, self.train_target, self.train_input_flat, self.train_labels = self.prepareDatasetForAutoencoder(train_set[0], train_set[1])
        self.valid_input, self.valid_target, self.valid_input_flat, self.valid_labels = self.prepareDatasetForAutoencoder(valid_set[0], valid_set[1])
        self.test_input, self.test_target, self.test_input_flat, self.test_labels = self.prepareDatasetForAutoencoder(test_set[0], test_set[1])
#         self.train_input = self.train_input[0:1000]
#         self.train_target = self.train_target[0:1000]
#         self.train_labels = self.train_labels[0:1000]
#         self.train_input_flat = self.train_input_flat[0:1000]
        
    def prepareDatasetForAutoencoder(self, inputs, targets):
        X = inputs
        X = X.reshape((-1, 1, 28, 28))*256*0.02
        return (X, X, X.reshape((-1, 28 * 28)), targets)

    def iterate_minibatches(self, set_type, batch_size, shuffle=False):
        inputs = None
        targets = None
        if set_type == 'train':
            inputs = self.train_input
            targets = self.train_target
        elif set_type == 'train_flat':
            inputs = self.train_input_flat
            targets = self.train_input_flat
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

def rescaleReshapeAndSaveImage(image_sample, out_filename):
    image_sample = ((image_sample - np.amin(image_sample)) /(np.amax(image_sample) - np.amin(image_sample))) * 255;
    image_sample = np.rint(image_sample).astype(int)
    image_sample = np.clip(image_sample, a_min=0, a_max=255).astype('uint8')
    img = Image.fromarray(image_sample, 'L')
    img.save(out_filename)
