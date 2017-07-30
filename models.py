'''
Created on Jul 21, 2017

@author: yawarnihal
'''

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
'use_inverse_layers': False,
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
'use_inverse_layers': False,
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
'use_inverse_layers': False,
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

arch8 = {
'use_inverse_layers': False,
'name': 'c-3-32_p_c-3-64_p_fc-10',
'layers_encode': [
        {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 32,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 64,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Encode',
        'encode_size':10,
        'non_linearity': 'rectify'
        },
    ]
}


arch7 = {
'use_inverse_layers': False,
'name': 'c-3-32_p_c-3-64_p_fc-32',
'layers_encode': [
        {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 32,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 64,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Encode',
        'encode_size':32,
        'non_linearity': 'rectify'
        },
    ]
}

arch9 = {
'use_inverse_layers': False,
'name': 'c-3-32_p_fc-32',
'layers_encode': [
        {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 32,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Encode',
        'encode_size':32,
        'non_linearity': 'rectify'
        },
    ]
}

arch10 = {
'use_inverse_layers': False,
'name': 'c-3-32_p_c-3-64_p_fc-128',
'layers_encode': [
        {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 32,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 64,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Encode',
        'encode_size':128,
        'non_linearity': 'rectify'
        },
    ]
}

arch11 = {
'use_inverse_layers': False,
'name': 'c-3-32_p_c-3-64_p_fc-64',
'layers_encode': [
        {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 32,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 64,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Encode',
        'encode_size':64,
        'non_linearity': 'rectify'
        },
    ]
}

arch12 = {
'use_inverse_layers': False,
'name': 'c-3-32_p_c-3-64_p_c-3-128_p_fc-32',
'layers_encode': [
        {
        'layer_type':'Input',
        'output_shape': [1, 28, 28]
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 32,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Conv2D',
        'num_filters': 64,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
	},
        {
        'layer_type':'Conv2D',
        'num_filters': 128,
        'filter_size': (3, 3),
        'non_linearity': 'rectify'
        },
        {
        'layer_type':'MaxPool2D',
        'filter_size': (2, 2),
        },
        {
        'layer_type':'Encode',
        'encode_size':32,
        'non_linearity': 'rectify'
        },
    ]
}


