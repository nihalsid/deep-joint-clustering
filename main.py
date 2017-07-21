'''
Created on Jul 9, 2017
@author: yawarnihal, eliealjalbout
'''
import time

import numpy

from misc import Dataset, rescaleReshapeAndSaveImage, evaluateKMeans
import models
from network import DCJC, rootLogger


def testOnlyConvAutoEncoder(arch):
    rootLogger.info("Loading dataset")
    dataset = Dataset()
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
            inputs, targets = batch
            err = dcjc.validate(inputs, targets)
            validation_error += err
            validation_batch_count += 1
        rescaleReshapeAndSaveImage(dcjc.predictReconstruction([dataset.train_input[test_image_index]])[0][0], 'outputs/' + str(epoch + 1) + '.png')
        rootLogger.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        rootLogger.info("  training loss:\t\t{:.6f}".format(train_error / train_batch_count))
        rootLogger.info("  validation loss:\t\t{:.6f}".format(validation_error / validation_batch_count))

def testOnlyClusterInitialization(arch, epochs):
    rootLogger.info("Loading dataset")
    dataset = Dataset()
    dataset.loadDataSet()
    rootLogger.info("Done loading dataset")
    rootLogger.info("Creating network")
    dcjc = DCJC(arch)
    rootLogger.info("Done creating network")
    rootLogger.info("Starting training")
    dcjc.pretrainWithData(dataset, epochs);
    
def testKMeans(methods):
    rootLogger.info('Initial Cluster Quality Comparison')
    rootLogger.info(60 * '_')
    rootLogger.info('%-30s     %8s     %8s' % ('method', 'ARI', 'AMI'))
    rootLogger.info(60 * '_') 
    dataset = Dataset()
    dataset.loadDataSet()
    #rootLogger.info(evaluateKMeans(dataset.train_input_flat, dataset.train_labels, 'image')[0])
    for m in methods:
        Z = numpy.load('models/z_' + m['name'] + '.npy')
        rootLogger.info(evaluateKMeans(Z, dataset.train_labels, m['name'])[0])
    rootLogger.info(60 * '_') 
    
def testOnlyClusterImprovement(arch, epochs, repeats): 
    rootLogger.info("Loading dataset")
    dataset = Dataset()
    dataset.loadDataSet()
    rootLogger.info("Done loading dataset")
    rootLogger.info("Creating network")
    dcjc = DCJC(arch)
    rootLogger.info("Starting cluster improvement")
    dcjc.doClustering(dataset, True, epochs, repeats)
    
if __name__ == '__main__':
#    testOnlyClusterInitialization(arch5, 1)
#     testOnlyClusterInitialization(arch7, 100)
#     testOnlyClusterInitialization(arch4, 1) 
#     testOnlyClusterInitialization(arch3, 1)    
#     testOnlyClusterInitialization(arch2, 1)
#     testOnlyClusterInitialization(arch0, 50)
    testKMeans([models.arch7])  # ([arch5, arch6, arch4, arch3, arch2, arch0])
    testOnlyClusterImprovement(models.arch7, 1, 20)
