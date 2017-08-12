import numpy as np

hyper = {
  "use"      : "adam",
  # bn ?
  "cifar10"  : {
    "epochs"         : 201,
    "sample_rate"    : 10,
    "sub_sample"     : [10, 30],
    "sub_sample_rate": [4, 2, 1],
    "logger_level"   : "info",
    "dbg"            : 0,
    "grids"          : {
      'dataset'   : ['cifar10', ],
      'model_type': ['vgg6'],  # 'vgg16', , 'vgg19'
      'lr'        : np.logspace(-2, -3, 2),
      'runtime'   : [1, ],
      'with_bn'   : [True, False]
    }
  },
  # mnist ok ?
  "mnist"    : {
    "epochs"         : 201,
    "sample_rate"    : 10,
    "sub_sample"     : [
      10,
      30
    ],
    "sub_sample_rate": [4, 2, 1],
    "logger_level"   : "info",
    "dbg"            : 0
  },
  # adam sgd
  "adam"     : {
    "epochs"         : 201,
    "sample_rate"    : 10,
    "sub_sample"     : [10, 30],
    "sub_sample_rate": [4, 2, 1],
    "logger_level"   : "info",
    "dbg"            : 0,
    "grids"          : {
      'model_type': ['vgg10', ],  # 'resnet10', ]  # 'vgg8', 'resnet8','vgg6', 'resnet6',
      'lr'        : np.logspace(-2, -3, 2),
      'optimizer' : ['rmsprop', 'adam'],
    }
  },
  # lr search
  "lr_search": {
    'dataset'    : ['cifar10', 'cifar100'],
    'model_type' : ['vgg10', 'resnet10', ],  # 'vgg8', 'resnet8','vgg6', 'resnet6',
    'lr'         : np.logspace(-2, -3, 4),
    'decay_epoch': [35, 75, 105],
    'decay'      : [2, 5, 10, 20],
  }
  
}
