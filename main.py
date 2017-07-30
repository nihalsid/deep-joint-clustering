'''
Created on Jul 9, 2017
@author: yawarnihal, eliealjalbout
'''
import time

import numpy

from misc import Dataset, rescaleReshapeAndSaveImage, evaluateKMeans,visualizeData
import models
from network import DCJC, rootLogger


def testOnlyConvAutoEncoder(arch):
    
    rootLogger.info("Loading dataset")
    dataset = Dataset('MNIST','mnist/mnist.pkl.gz',10)
    dataset.loadDataSet()
    test_image_index = 2
    rescaleReshapeAndSaveImage(dataset.train_input[test_image_index][0], 'outputs/input_' + str(test_image_index) + '.png')
    rootLogger.info("Done loading dataset")
    rootLogger.info("Creating network")
    dcjc = DCJC(arch)
    rootLogger.info("Done creating network")
    rootLogger.info("Starting training")
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
            inputs, targets = batchtestOnlyClusterInitialization
            err = dcjc.validate(inputs, targets)
            validation_error += err
            validation_batch_count += 1
        rescaleReshapeAndSaveImage(dcjc.predictReconstruction([dataset.train_input[test_image_index]])[0][0], 'outputs/' + str(epoch + 1) + '.png')
        rootLogger.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        rootLogger.info("  training loss:\t\t{:.6f}".format(train_error / train_batch_count))
        rootLogger.info("  validation loss:\t\t{:.6f}".format(validation_error / validation_batch_count))

def testOnlyClusterInitialization(arch, epochs,plot=[False,False]):
    
    rootLogger.info("Loading dataset")
    dataset = Dataset('MNIST','mnist/mnist.pkl.gz',10)
    dataset.loadDataSet()
    rootLogger.info("Done loading dataset")

    if plot[0]:
        visualizeData(dataset.train_input,dataset.train_labels,dataset.cluster_nrs)

    rootLogger.info("Creating network")
    dcjc = DCJC(arch)
    rootLogger.info("Done creating network")
    
    rootLogger.info("Starting training")
    dcjc.pretrainWithData(dataset, epochs,plot[1]);
    
def testKMeans(methods):
    
    rootLogger.info('Initial Cluster Quality Comparison')
    rootLogger.info(60 * '_')
    rootLogger.info('%-30s     %8s     %8s' % ('method', 'ARI', 'AMI'))
    rootLogger.info(60 * '_') 
    dataset = Dataset('MNIST','mnist/mnist.pkl.gz',10)
    dataset.loadDataSet()
    #rootLogger.info(evaluateKMeans(dataset.train_input_flat, dataset.train_labels, 'image')[0])
    for m in methods:
        Z = numpy.load('models/z_' + m['name'] + '.npy')
        rootLogger.info(evaluateKMeans(Z, dataset.train_labels, m['name'])[0])
    rootLogger.info(60 * '_') 
    
def testOnlyClusterImprovement(arch, epochs, repeats,plot=[False,False]): 
    
    rootLogger.info("Loading dataset")
    dataset = Dataset('MNIST','mnist/mnist.pkl.gz',10)
    dataset.loadDataSet()
    rootLogger.info("Done loading dataset")

    if plot[0]:
        visualizeData(dataset.train_input,dataset.train_labels,dataset.cluster_nrs)
    
    rootLogger.info("Creating network")
    dcjc = DCJC(arch)
    
    rootLogger.info("Starting cluster improvement")
    dcjc.doClustering(dataset, True, epochs, repeats,plot[1])

    
if __name__ == '__main__':
    
    #testOnlyClusterInitialization(models.arch0, 500,plot=[False,True])
    #testOnlyClusterInitialization(models.arch7, 500,plot=[False,True])
    #testOnlyClusterInitialization(models.arch8, 500,plot=[False,True])
    #testOnlyClusterInitialization(models.arch9, 500,plot=[False,True])    
    #testKMeans([models.arch0, models.arch7, models.arch8, models.arch9])
    testOnlyClusterImprovement(models.arch7, 20, 1,plot=[False,True])
