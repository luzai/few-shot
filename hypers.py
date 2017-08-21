import numpy as np

hyper = {
  "gpu"         : [0, 1, 2, 3],
  "use"         : "cifar",
  "logger_level": "info",
  "dbg"         : False,
  
  "log_stat"    : True,
  # we need 2 mode : log all stat
  # stat and tensor for only Layer
  "log_tensor"  : True,
  "last_only" : True,
  "cifar"       : {
    "epochs"         : 71,
    "sample_rate"    : 1.,  # epoch^-1
    "sub_sample"     : [3, 5],  # epoch
    "sub_sample_rate": [10, 5, 1],
    "grids"          : {'dataset'   : ['cifar10', ],
                        'model_type': ['resnet10', 'vgg10', ],
                        'optimizer' : [
                          # {'name': 'sgd',
                          #               'lr'  : 0.001, },
                          {'name': 'sgd',
                           'lr'  : 0.01, },
                          # {'name'       : 'sgd',
                          #  'lr'         : 0.01,
                          #  'decay_epoch': [150, ],
                          #  'decay'      : [10, ], },
                          # {'name'       : 'sgd',
                          #  'lr'         : 0.01,
                          #  'decay_epoch': [50, ],
                          #  'decay'      : [10, ], },
                        ],
                        'hiddens'   : [5, 20, 512]
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
                        }
  },
  'vgg'         : {
    "epochs": 61,
    "grids" : {'dataset'   : ['cifar10', ],  # 'cifar10'],
               'model_type': ['vgg101', 'vgg102', 'vgg103', 'resnet10'],  # 'vgg16', , 'vgg19'
               'optimizer' : [{'name': 'sgd',
                               'lr'  : 0.01, },
                              ],
               'hiddens'   : [5, 20, 512]
               }
  },
  
  "win_size"    : 31,
  # "use_bias"    : False,  # todo
  "curve_only"  : False,
}
