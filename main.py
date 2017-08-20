'''
Created on Jul 9, 2017
@author: yawarnihal, eliealjalbout
'''
import numpy
import json
from misc import DatasetHelper, evaluateKMeans, visualizeData
from network import DCJC, rootLogger

def testOnlyClusterInitialization(dataset_name, arch, epochs):
    
    rootLogger.info("Loading dataset")
    dataset = DatasetHelper(dataset_name)
    dataset.loadDataset()
    rootLogger.info("Done loading dataset")
    rootLogger.info("Creating network")
    dcjc = DCJC(arch)
    rootLogger.info("Done creating network")
    
    rootLogger.info("Starting training")
    dcjc.pretrainWithData(dataset, epochs);
    
def testKMeans(dataset_name, methods):
    rootLogger.info('Initial Cluster Quality Comparison')
    rootLogger.info(60 * '_')
    rootLogger.info('%-30s     %8s     %8s' % ('method', 'ARI', 'AMI'))
    rootLogger.info(60 * '_') 
    dataset = DatasetHelper(dataset_name)
    dataset.loadDataset()
    #rootLogger.info(evaluateKMeans(dataset.train_input_flat, dataset.train_labels, dataset.getClusterCount(), 'image')[0])
    for m in methods:
        Z = numpy.load('saved_params/' + dataset.name + '/z_' + m['name'] + '.npy')
        rootLogger.info(evaluateKMeans(Z, dataset.train_labels, dataset.getClusterCount(), m['name'])[0])
    rootLogger.info(60 * '_') 
    
def testOnlyClusterImprovement(dataset_name, arch, epochs, repeats):
    
    rootLogger.info("Loading dataset")
    dataset = DatasetHelper(dataset_name)
    dataset.loadDataset()
    rootLogger.info("Done loading dataset")

    rootLogger.info("Creating network")
    dcjc = DCJC(arch)

    rootLogger.info("Starting cluster improvement")
    dcjc.doClustering(dataset, True, epochs, repeats)

    
if __name__ == '__main__':
    mnist_models = []
    coil_models = []
    with open("models/coil.json") as models_file:
        coil_models = json.load(models_file)
    with open("models/mnist.json") as models_file:
        mnist_models = json.load(models_file)
    testOnlyClusterInitialization('MNIST', mnist_models[1], 50)
    # testOnlyClusterInitialization(mnist_models.arch0, 500,plot=[False,True])
    # testOnlyClusterInitialization(mnist_models.arch7, 500,plot=[False,True])
    # testOnlyClusterInitialization(mnist_models.arch8, 500,plot=[False,True])
    # testOnlyClusterInitialization(mnist_models.arch9, 500,plot=[False,True])    
    # testKMeans([mnist_models.arch0, mnist_models.arch7, mnist_models.arch8, mnist_models.arch9])
    # testOnlyClusterImprovement('COIL20',coil_models.arch7, 1, 20, plot=[False,False])
