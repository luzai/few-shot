import numpy as np

hyper = {
  "use"         : "mnist",
  "logger_level": "info",
  "dbg"         : False,
  # bn ?
  "cifar10"     : {
    "epochs"         : 151,
    "sample_rate"    : 0.1,  # epoch^-1
    "sub_sample"     : [10, 30],  # epoch
    "sub_sample_rate": [3, 2, 1],
    "grids"          : {
      'dataset'   : ['cifar10', 'cifar100'],
      'model_type': ['vgg6', 'vgg16', 'vgg19', 'resnet10'],
      'lr'        : np.logspace(-2, -3, 2),
      'with_bn'   : [True, False]
    }
  },
  # vgg depth: vgg19
  "vgg19"       : {
    "epochs"         : 201,
    "sample_rate"    : 1,  # epoch^-1
    "sub_sample"     : [10, 30],  # epoch
    "sub_sample_rate": [4, 2, 1],
    "grids"          : {
      'dataset'   : ['cifar10', ],
      'model_type': ['vgg16', 'vgg19'],  # 'vgg16', , 'vgg19'
      'lr'        : np.logspace(-2, -3, 2),
      'with_bn'   : [True]
    }
  },
  # mnist ok ?
  "mnist"       : {
    "epochs"         : 5,
    "sample_rate"    : 10,
    "sub_sample"     : [1, 2],
    "sub_sample_rate": [5, 3, 1],
    "grids"          : {
      'dataset'   : ['mnist', ],
      'model_type': ['vgg6'],  # 'vgg16', , 'vgg19'
      'lr'        : np.logspace(-2, -3, 2),
      'with_bn'   : [True, False]
    }
  },
  
  # adam sgd
  "adam"        : {
    "epochs": 201,
    "grids" : {
      'model_type': ['vgg10', ],  # 'resnet10', ]  # 'vgg8', 'resnet8','vgg6', 'resnet6',
      'optimizer' : ['rmsprop', 'adam'],
      'lr'        : np.logspace(-3, -4, 2)
    }
  },
  # lr search
  "lr_search"   : {
    "epochs": 201,
    'grids' : {
      'dataset'    : ['cifar10'],  # , 'cifar100'
      'model_type' : ['vgg10', 'resnet10', ],  # 'vgg8', 'resnet8','vgg6', 'resnet6',
      'lr'         : [0.1],
      'decay_epoch': [50, 125, 185],
      'decay'      : [10],
    }
  },
  # std time
  
}
