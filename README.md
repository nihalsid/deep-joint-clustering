Deep Learning for Clustering
===========================

This repository contains codes and data for a deep learning and clustering approach. It contains the following:

### Packages:
- models    : contains saved models with format m_.. or outputs with format z_.. .
- plots     : contains usefull plots of the obtained results such as data representation at several stages.
- tests     : contains test scripts we ran embedding in the code.
- logs      : contains debugging and run-time logs.

### Scripts:
- customlayer.py    : contains custom created layers using lasagne such as unpooling layer and a clustering layer that calculates the soft assignement probability.
- main.py           : contains the entry function to either learning or test of the neural network. By simple changes it is possible to  build and learn/load a network which satisfies different loss function or specific architecture.
- misc.py           : contains utility functions for a variety of purposes: visualisation, dataset management, mini-batch iteration, clustering evaluation.
- network.py        : contains class which creates a network based on a presented description map, and functions for training (in different ways) and output generation.


## Prerequesites:
- [Install Theano](http://deeplearning.net/software/theano/install.html)
- [Install Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html)
- Install Pickle : pip install pickle
- Install Scikit learn: pip install -U scikit-learn
- Install Cython (for tsne) : pip install cython
- Install tsne: pip install tsne
