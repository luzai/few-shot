import numpy as np

hyper = {
  "gpu"         : [0, 1, 2, 3],
  "use"         : "cifar10",
  "logger_level": "info",
  "dbg"         : True,
  "win_size"    : 11,
  "log_stat"   : True,
  "log_tensor":True,
  "cifar10"     : {
    "epochs"         : 201,
    "sample_rate"    : 1,  # epoch^-1
    "sub_sample"     : [9, 30],  # epoch
    "sub_sample_rate": [65, 4, 1],
    "grids"          : {'dataset'   : ['cifar10'],
                        'model_type': ['resnet10', 'vgg10', ],
                        'optimizer' : [
                          {'name': 'sgd',
                           'lr'  : 0.001, },
                          {'name': 'sgd',
                           'lr'  : 0.01, },
                          # {'name': 'sgd',
                          #  'lr'         : 0.01,
                          #  'decay_epoch': [150, ],
                          #  'decay'      : [10, ], },
                          # {'name'       : 'sgd',
                          #  'lr'         : 0.01,
                          #  'decay_epoch': [50, ],
                          #  'decay'      : [10, ], },
                        ],
                        }
  },
  # mnist ok
  "mnist"       : {
    "epochs"         : 5,
    "sample_rate"    : 10,
    "sub_sample"     : [1, 2],
    "sub_sample_rate": [5, 3, 1],
    "grids"          : {'dataset'   : ['mnist', ],  # 'cifar10'],
                        'model_type': ['resnet10'],  # 'vgg16', , 'vgg19'
                        'optimizer' : [{'name': 'sgd',
                                        'lr'  : 0.001,
                                        },
                                       {'name': 'sgd',
                                        'lr'  : 0.01,
                                        },
                                       ],
                        'with_bn'   : [True]
                        }
  },
  
 
}
