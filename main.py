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
    dcjc.pretrainWithData(dataset, epochs, False);
    
def testKMeans(dataset_name, methods):
    rootLogger.info('Initial Cluster Quality Comparison')
    rootLogger.info(80 * '_')
    rootLogger.info('%-50s     %8s     %8s' % ('method', 'ACC', 'NMI'))
    rootLogger.info(80 * '_')
    dataset = DatasetHelper(dataset_name)
    dataset.loadDataset()
    # rootLogger.info(evaluateKMeans(dataset.input_flat, dataset.labels, dataset.getClusterCount(), 'image')[0])
    for m in methods:
        Z = numpy.load('saved_params/' + dataset.name + '/z_' + m['name'] + '.npy')
        rootLogger.info(evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), m['name'])[0])
    rootLogger.info(80 * '_')
    
def testOnlyClusterImprovement(dataset_name, arch, epochs, method):
    
    rootLogger.info("Loading dataset")
    dataset = DatasetHelper(dataset_name)
    dataset.loadDataset()
    rootLogger.info("Done loading dataset")

    rootLogger.info("Creating network")
    dcjc = DCJC(arch)

    rootLogger.info("Starting cluster improvement")
    if method == 'KM':
        dcjc.doClusteringWithKMeansLoss(dataset, epochs)
    elif method == 'KLD':
        dcjc.doClusteringWithKLdivLoss(dataset, True, epochs)

def visualizeLatentSpace(dataset_name, arch):
    rootLogger.info("Loading dataset")
    dataset = DatasetHelper(dataset_name)
    dataset.loadDataset()
    rootLogger.info("Done loading dataset")
    visualizeData(dataset.input_flat[0:5000], dataset.labels[0:5000], dataset.getClusterCount(), "plots/%s/raw.png"%dataset.name)
    Z = numpy.load('saved_params/' + dataset.name + '/z_' + arch['name'] + '.npy')
    visualizeData(Z[0:5000], dataset.labels[0:5000], dataset.getClusterCount(), "plots/%s/autoencoder.png"%dataset.name)
    Z = numpy.load('saved_params/' + dataset.name + '/pc_z_' + arch['name'] + '.npy')
    visualizeData(Z[0:5000], dataset.labels[0:5000], dataset.getClusterCount(), "plots/%s/clustered.png"%dataset.name)

if __name__ == '__main__':
    mnist_models = []
    coil_models = []
    with open("models/coil.json") as models_file:
        coil_models = json.load(models_file)
    with open("models/mnist.json") as models_file:
        mnist_models = json.load(models_file)
    #visualizeLatentSpace('COIL20', coil_models[0])
    visualizeLatentSpace('MNIST', mnist_models[1])
    # testOnlyClusterInitialization('MNIST', mnist_models[1], 500)
    # testOnlyClusterImprovement('MNIST', mnist_models[1], 100, 'KLD')
    # testOnlyClusterInitialization(mnist_models.arch0, 500,plot=[False,True])
    # testOnlyClusterInitialization(mnist_models.arch7, 500,plot=[False,True])
    # testOnlyClusterInitialization(mnist_models.arch8, 500,plot=[False,True])
    # testOnlyClusterInitialization(mnist_models.arch9, 500,plot=[False,True])
    # testKMeans("COIL20",[coil_models[0]])
    # testOnlyClusterImprovement('COIL20',coil_models.arch7, 1, 20, plot=[False,False])
