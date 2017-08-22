'''
Created on Jul 11, 2017

@author: yawarnihal, eliealjalbout
'''

import cPickle
import gzip

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy import float32
from sklearn import metrics
from sklearn.cluster.k_means_ import KMeans
from tsne import bh_sne
from external.purity import purity_score

class DatasetHelper(object):
    def __init__(self, name):
        self.name = name
        if name == 'MNIST':
            self.dataset = MNISTDataset()
        elif name == 'STL':
            self.dataset = STLDataset()
        elif name == 'COIL20':
            self.dataset = COIL20Dataset()

    def loadDataset(self):
        self.input, self.labels, self.input_flat = self.dataset.loadDataset()

    #         self.input = self.input[0:1000]
    #         self.labels = self.labels[0:1000]
    #         self.input_flat = self.input_flat[0:1000]

    def getClusterCount(self):
        return self.dataset.cluster_count

    def iterate_minibatches(self, set_type, batch_size, targets=None, shuffle=False):
        inputs = None
        if set_type == 'IMAGE':
            inputs = self.input
            if targets is None:
                targets = self.input
        elif set_type == 'FLAT':
            inputs = self.input_flat
            if targets is None:
                targets = self.input_flat
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


class MNISTDataset(object):
    def __init__(self):
        self.cluster_count = 10

    def loadDataset(self):
        f = gzip.open('mnist/mnist.pkl.gz', 'rb')
        train_set, _, test_set = cPickle.load(f)
        train_input, train_input_flat, train_labels = self.prepareDatasetForAutoencoder(train_set[0], train_set[1])
        test_input, test_input_flat, test_labels = self.prepareDatasetForAutoencoder(test_set[0], test_set[1])
        f.close()
        return [np.concatenate((train_input, test_input)), np.concatenate((train_labels, test_labels)),
                np.concatenate((train_input_flat, test_input_flat))]

    def prepareDatasetForAutoencoder(self, inputs, targets):
        X = inputs
        X = X.reshape((-1, 1, 28, 28))
        return (X, X.reshape((-1, 28 * 28)), targets)


class STLDataset(object):
    def __init__(self):
        self.cluster_count = 10

    def loadDataset(self):
        train_x = np.fromfile('stl/train_X.bin', dtype=np.uint8)
        train_y = np.fromfile('stl/train_y.bin', dtype=np.uint8)
        test_x = np.fromfile('stl/train_X.bin', dtype=np.uint8)
        test_y = np.fromfile('stl/train_y.bin', dtype=np.uint8)
        train_input = np.reshape(train_x, (-1, 3, 96, 96))
        train_labels = train_y
        train_input_flat = np.reshape(test_x, (-1, 1, 3 * 96 * 96))
        test_input = np.reshape(test_x, (-1, 3, 96, 96))
        test_labels = test_y
        test_input_flat = np.reshape(test_x, (-1, 1, 3 * 96 * 96))
        return [np.concatenate(train_input, test_input), np.concatenate(train_labels, test_labels),
                np.concatenate(train_input_flat, test_input_flat)]


class COIL20Dataset(object):
    def __init__(self):
        self.cluster_count = 20

    def loadDataset(self):
        train_x = np.load('coil/coil_X.npy').astype(np.float32) / 256.0
        train_y = np.load('coil/coil_y.npy')
        train_x_flat = np.reshape(train_x, (-1, 128 * 128))
        return [train_x, train_y, train_x_flat]


def rescaleReshapeAndSaveImage(image_sample, out_filename):
    image_sample = ((image_sample - np.amin(image_sample)) / (np.amax(image_sample) - np.amin(image_sample))) * 255;
    image_sample = np.rint(image_sample).astype(int)
    image_sample = np.clip(image_sample, a_min=0, a_max=255).astype('uint8')
    img = Image.fromarray(image_sample, 'L')
    img.save(out_filename)


def getClusterMetricString(method_name, labels_true, labels_pred):
    return '%-50s     %8.3f     %8.3f' % (method_name, purity_score(labels_true, labels_pred),
                                          metrics.normalized_mutual_info_score(labels_true, labels_pred))


def evaluateKMeans(data, labels, nclusters, method_name):
    kmeans = KMeans(n_clusters=nclusters, n_init=20)
    kmeans.fit(data)
    return getClusterMetricString(method_name, labels, kmeans.labels_), kmeans.cluster_centers_


def visualizeData(data, clust_assig=None, cluster_nrs=-1):
    # convert image data to float64 matrix. float64 is need for bh_sne
    data = np.asarray(data).astype('float32')
    data = data.reshape((data.shape[0], -1))

    # get vizualization data and split to x and y
    vis_data = bh_sne(data)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    # show data
    if (clust_assig != None):  # if predicted assignements exist then color code can be used to show it
        plt.scatter(vis_x, vis_y, c=clust_assig, cmap=plt.cm.get_cmap("jet", cluster_nrs))
        plt.colorbar(ticks=range(cluster_nrs))
        plt.clim(-0.5, 9.5)
    else:
        plt.plot(vis_x, vis_y)

    plt.show()
