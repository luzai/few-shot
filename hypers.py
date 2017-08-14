import numpy as np

hyper = {
  "gpu"         : [0, 1, 2, 3],
  "use"         : "lr_search",
  "logger_level": "info",
  "dbg"         : True,
  # bn ?
  "cifar10"     : {
    "epochs"         : 231,
    "sample_rate"    : 1 / 4.,  # epoch^-1
    "sub_sample"     : [10, 30],  # epoch
    "sub_sample_rate": [3, 2, 1],
    "grids"          : {
      'dataset'   : ['cifar10', 'cifar100'],
      'model_type': ['resnet10', 'resnet18', 'vgg10', 'vgg19'],
      'lr'        : np.logspace(-2, -3, 2),
      'with_bn'   : [True, False]
    }
  },
  # mnist ok
  "mnist"       : {
    "epochs"         : 5,
    "sample_rate"    : 10,
    "sub_sample"     : [1, 2],
    "sub_sample_rate": [5, 3, 1],
    "grids"          : {
      'dataset'   : ['mnist', 'cifar10'],
      'model_type': ['vgg6'],  # 'vgg16', , 'vgg19'
      'lr'        : np.logspace(-2, -3, 2),
      'with_bn'   : [True]
    }
  },
  # lr search
  "lr_search"   : {
    "epochs"         : 201,
    "sample_rate"    : 1 / 4.,  # epoch^-1
    "sub_sample"     : [10, 30],  # epoch
    "sub_sample_rate": [3, 2, 1],
    'grids'          : {
      'dataset'   : ['cifar10'],  # , 'cifar100'
      'model_type': ['resnet10', ],  # 'vgg8', 'resnet8','vgg6', 'resnet6',
      'optimizer' : [{'name'       : 'sgd',
                      'lr'         : 0.01,
                      'decay_epoch': [10, 75, 125],
                      'decay'      : [0.1, 10, 10], },
                     {'name': 'adam',
                      'lr'  : 0.001},
                     # {'name' : 'sgd',
                     #  'decay': 'cos'}
                     ]
    }
  },
}

if __name__ == '__main__':
  import sys
  
  sys.path.append('./src')
  from vis_utils import cartesian
  
  tmp = {
    'dataset'   : ['cifar10'],  # , 'cifar100'
    'model_type': ['resnet10', ],  # 'vgg8', 'resnet8','vgg6', 'resnet6',
    'optimizer' : [{'name'       : 'sgd',
                    'lr'         : 0.01,
                    'decay_epoch': [10, 75, 125],
                    'decay'      : [0.1, 10, 10], },
                   {'name': 'adam',
                    'lr'  : 0.001},
                   # {'name' : 'sgd',
                   #  'decay': 'cos'}
                   ]
  }
  import pprint
  
  pprint.pprint(cartesian(tmp.values()))
