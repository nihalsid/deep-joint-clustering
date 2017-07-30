'''
Created on Jul 11, 2017

@author: yawarnihal, eliealjalbout
'''

import cPickle
import gzip

from PIL import Image
from sklearn import metrics
from sklearn.cluster.k_means_ import KMeans
from matplotlib import pyplot as plt
from tsne import bh_sne

import numpy as np

# todo: write function that counts cluster numbers
class Dataset(object):

    def __init__(self
                ,name
                ,fname
                ,cluster_nrs
                ):

        self.name           = name
        self.fname          = fname
        self.cluster_nrs    = cluster_nrs

    
    def loadDataSet(self):
        
        f = gzip.open(self.fname, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        self.train_input, self.train_target, self.train_input_flat, self.train_labels = self.prepareDatasetForAutoencoder(train_set[0], train_set[1])
        self.valid_input, self.valid_target, self.valid_input_flat, self.valid_labels = self.prepareDatasetForAutoencoder(valid_set[0], valid_set[1])
        self.test_input, self.test_target, self.test_input_flat, self.test_labels = self.prepareDatasetForAutoencoder(test_set[0], test_set[1])
        self.train_input = np.concatenate((self.train_input, self.valid_input, self.test_input))
        self.train_target = np.concatenate((self.train_target, self.valid_target, self.test_target))
        self.train_labels = np.concatenate((self.train_labels, self.valid_labels, self.test_labels))
        self.train_input_flat = np.concatenate((self.train_input_flat, self.valid_input_flat, self.test_input_flat))
#         self.train_input = self.train_input[0:10000]
#         self.train_target = self.train_target[0:10000]
#         self.train_labels = self.train_labels[0:10000]
#         self.train_input_flat = self.train_input_flat[0:10000]
        
    def prepareDatasetForAutoencoder(self, inputs, targets):
        
        X = inputs
        X = X.reshape((-1, 1, 28, 28)) 
        return (X, X, X.reshape((-1, 28 * 28)), targets)

    def iterate_minibatches(self, set_type, batch_size, targets=None, shuffle=False):
        
        inputs = None
        if set_type == 'train':
            inputs = self.train_input
            if targets is None:
                targets = self.train_target
        elif set_type == 'train_flat':
            inputs = self.train_input_flat
            if targets is None:
                targets = self.train_input_flat
                
        elif set_type == 'validation':
            inputs = self.valid_input
            if targets is None:
                targets = self.valid_target
        elif set_type == 'test':
            inputs = self.test_input
            if targets is None:
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
    
    image_sample = ((image_sample - np.amin(image_sample)) / (np.amax(image_sample) - np.amin(image_sample))) * 255;
    image_sample = np.rint(image_sample).astype(int)
    image_sample = np.clip(image_sample, a_min=0, a_max=255).astype('uint8')
    img = Image.fromarray(image_sample, 'L')
    img.save(out_filename)

def getClusterMetricString(method_name, labels_true, labels_pred):
    
    return '%-30s     %8.3f     %8.3f' % (method_name, metrics.adjusted_rand_score(labels_true, labels_pred), metrics.adjusted_mutual_info_score(labels_true, labels_pred))

def evaluateKMeans(data, labels, method_name):
    
    kmeans = KMeans(n_clusters=10, n_init=20)
    kmeans.fit(data)
    return getClusterMetricString(method_name, labels, kmeans.labels_), kmeans.cluster_centers_


def visualizeData(data,clust_assig=None, cluster_nrs=-1):

    # convert image data to float64 matrix. float64 is need for bh_sne
    data = np.asarray(data).astype('float64')
    data = data.reshape((data.shape[0], -1))

    # get vizualization data and split to x and y
    vis_data = bh_sne(data)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]


    # show data
    if(clust_assig!=None): # if predicted assignements exist then color code can be used to show it
        plt.scatter(vis_x, vis_y, c=clust_assig, cmap=plt.cm.get_cmap("jet", cluster_nrs))
        plt.colorbar(ticks=range(cluster_nrs))
        plt.clim(-0.5, 9.5)
    else:
        plt.plot(vis_x,vis_y)
        
    plt.show()