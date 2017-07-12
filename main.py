'''
Created on Jul 9, 2017
@author: yawarnihal, eliealjalbout
'''
import time
from misc import Dataset, rescaleReshapeAndSaveImage
from network import DCJC

arch1 = {
# a small conv, max pool, same small conv, max pool
# Loss after 20 epochs = 19.5
'use_inverse_layers': True,
'layers_encode': [
      {
        'layer_type':'input',
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
# architecture 1 with more filters in both convolutional layers, but same filter size
# Train loss after 20 epochs = 12.2
'use_inverse_layers': True,
'layers_encode': [
      {
        'layer_type':'input',
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
# Half of arch 1
# Train loss after 20 epochs= 0.07
'use_inverse_layers': True,
'layers_encode': [
      {
        'layer_type':'input',
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
# Train loss after 10 iterations - 0.04
'use_inverse_layers': True,
'layers_encode': [
      {
        'layer_type':'input',
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
'use_inverse_layers': True,
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
        'encode_size':4,
        },
    ]
}

if __name__=="__main__":
    print("\nLoading dataset...")
    dataset = Dataset()
    dataset.loadDataSet()
    test_image_index = 2
    rescaleReshapeAndSaveImage(dataset.train_input[test_image_index][0],dataset.train_mu,dataset.train_sigma, 'outputs/input_'+str(test_image_index)+'.png')
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
        rescaleReshapeAndSaveImage(dcjc.predict([dataset.train_input[test_image_index]])[0][0],dataset.train_mu,dataset.train_sigma, 'outputs/'+str(epoch+1)+'.png')
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_error / train_batch_count))
        print("  validation loss:\t\t{:.6f}".format(validation_error / validation_batch_count))
