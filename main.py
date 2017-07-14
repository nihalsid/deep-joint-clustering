'''
Created on Jul 9, 2017
@author: yawarnihal, eliealjalbout
'''
import time

import numpy
from sklearn import metrics
from sklearn.cluster.k_means_ import KMeans

from misc import Dataset, rescaleReshapeAndSaveImage
from network import DCJC


arch0 = {
    'use_inverse_layers': False,
    'name': 'fc-500_fc-500_fc-2000_fc-10',
    'layers_encode': [
        {
            'layer_type':'Input',
            'output_shape': [1, 1, 28 * 28]
        },
        {
            'layer_type':'Dense',
            'num_units': 500,
        },
        {
            'layer_type':'Dense',
            'num_units': 500,
        },
        {
            'layer_type':'Dense',
            'num_units': 2000,
        },
        {
            'layer_type':'Encode',
            'num_units': 10,
            'non_linearity': 'linear'
        },
    ]
} 

arch1 = {
'use_inverse_layers': True,
'name': 'c-6-5_p_c-6-5_p',
'layers_encode': [
      {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
      {
        'layer_type':'Conv2D',
        'num_filters': 6,
        'filter_size': (5, 5),
        'non_linearity': 'rectify'
        },
       {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
       {
        'layer_type':'Conv2D',
        'num_filters': 6,
        'filter_size': (5, 5),
        'non_linearity': 'rectify'
        },
       {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
    ]
}

arch2 = {
'use_inverse_layers': True,
'name':'c-5-32_p_c-5-32_p',
'layers_encode': [
      {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
      {
        'layer_type':'Conv2D',
        'num_filters': 32,
        'filter_size': (5, 5),
        'non_linearity': 'rectify'
        },
       {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
       {
        'layer_type':'Conv2D',
        'num_filters': 32,
        'filter_size': (5, 5),
        'non_linearity': 'rectify'
        },
       {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
    ]
}

arch3 = {
'use_inverse_layers': True,
'name':'c-6-5_p',
'layers_encode': [
      {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
      {
        'layer_type':'Conv2D',
        'num_filters': 6,
        'filter_size': (5, 5),
        'non_linearity': 'rectify'
        },
       {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
    ]
}


arch4 = {
# Half of arch 2
'use_inverse_layers': True,
'name': 'c-5-32_p',
'layers_encode': [
      {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
      {
        'layer_type':'Conv2D',
        'num_filters': 32,
        'filter_size': (5, 5),
        'non_linearity': 'rectify'
        },
       {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
    ]
}

arch5 = {
# Arch 4 with Fully connected encode layer
# Train accuracy = 10, irrespective of number of layers
'use_inverse_layers': True,
'name': 'c-5-32_p_fc-10',
'layers_encode': [
        {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 32,
        'filter_size': (5, 5),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Encode',
        'encode_size':10,
        'non_linearity': 'linear',
        },
    ]
}

arch6 = {
'use_inverse_layers': True,
'name': 'c-5-64_p_fc-10',
'layers_encode': [
        {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 64,
        'filter_size': (5, 5),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Encode',
        'encode_size':10,
        'non_linearity': 'linear'
        },
    ]
}

def testOnlyConvAutoEncoder():
    print("\nLoading dataset...")
    dataset = Dataset()
    dataset.loadDataSet()
    test_image_index = 2
    rescaleReshapeAndSaveImage(dataset.train_input[test_image_index][0], dataset.train_mu, dataset.train_sigma, 'outputs/input_' + str(test_image_index) + '.png')
    print("Done loading dataset\n")
    print("Creating network...")
    dcjc = DCJC(arch5)
    dcjc.printLayers()
    print("Done creating network\n")
    print("Starting training...")
    num_epochs = 50
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_error = 0
        train_batch_count = 0
        start_time = time.time()
        for batch in dataset.iterate_minibatches('train', 500, shuffle=True):
            inputs, targets = batch
            train_error += dcjc.train(inputs, targets)
            train_batch_count += 1
        validation_error = 0
        validation_batch_count = 0
        for batch in dataset.iterate_minibatches('validation', 500, shuffle=True):
            inputs, targets = batch
            err = dcjc.validate(inputs, targets)
            validation_error += err
            validation_batch_count += 1
        rescaleReshapeAndSaveImage(dcjc.predictReconstruction([dataset.train_input[test_image_index]])[0][0], dataset.train_mu, dataset.train_sigma, 'outputs/' + str(epoch + 1) + '.png')
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_error / train_batch_count))
        print("  validation loss:\t\t{:.6f}".format(validation_error / validation_batch_count))

def testClusterInitialization(arch, epochs):
    print("\nLoading dataset...")
    dataset = Dataset()
    dataset.loadDataSet()
    print("Done loading dataset\n")
    print("Creating network...")
    dcjc = DCJC(arch)
    dcjc.printLayers()
    print("Done creating network\n")
    print("Starting training...")
    dcjc.pretrainWithData(dataset, epochs);
    
def evaluateKMeans(data, labels, method_name):
    kmeans = KMeans(n_clusters=10, n_init=20)
    kmeans.fit(data)
    print '%-30s     %8.3f     %8.3f     %8.3f     %8.3f     %8.3f' % (method_name, metrics.homogeneity_score(labels, kmeans.labels_),
         metrics.completeness_score(labels, kmeans.labels_),
         metrics.v_measure_score(labels, kmeans.labels_),
         metrics.adjusted_rand_score(labels, kmeans.labels_),
         metrics.adjusted_mutual_info_score(labels, kmeans.labels_))

def testKMeans(methods):
    print 'Initial Cluster Quality Comparison'
    print(99 * '_')
    print('%-30s     %8s     %8s     %8s     %8s     %8s' % ('method', 'homo', 'compl', 'v-meas', 'ARI', 'AMI'))
    print(99 * '_') 
    dataset = Dataset()
    dataset.loadDataSet()
    # evaluateKMeans(dataset.train_input_flat, dataset.train_labels, 'image')
    for m in methods:
        Z = numpy.load('models/z_' + m['name'] + '.npy')
        evaluateKMeans(Z, dataset.train_labels, m['name'])
    
if __name__ == '__main__':
    testClusterInitialization(arch5, 1)
    testClusterInitialization(arch6, 1)
    testClusterInitialization(arch4, 1) 
    testClusterInitialization(arch3, 1)    
    testClusterInitialization(arch2, 1)
    testClusterInitialization(arch0, 2)
    testKMeans([arch5, arch6, arch4, arch3, arch2, arch0])
