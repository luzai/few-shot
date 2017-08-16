import numpy as np

hyper = {
  "gpu"         : [0, 1, 2, 3],
  "use"         : "cifar10",
  "logger_level": "info",
  "dbg"         : False,
  "win_size"    : 12,
  
  "cifar10_otho": {
    "epochs"         : 201,
    "sample_rate"    : 1. / 4,  # epoch^-1
    "sub_sample"     : [10, 30],  # epoch
    "sub_sample_rate": [3, 2, 1],
    "grids"          : {'dataset'   : ['cifar10'],
                        'model_type': ['resnet10', 'vgg10', ],
                        'optimizer' : [{'name': 'sgd',
                                        'lr'  : 0.001, },
                                       {'name': 'sgd',
                                        'lr'  : 0.01, },
                                       ],
                        'with_bn'   : [True]
                        }
  },
  
  "cifar10"     : {
    "epochs"         : 201,
    "sample_rate"    : 1. / 4,  # epoch^-1
    "sub_sample"     : [10, 30],  # epoch
    "sub_sample_rate": [3, 2, 1],
    "grids"          : {'dataset'   : ['cifar10'],
                        'model_type': ['resnet10', 'vgg10', ],
                        'optimizer' : [{'name': 'sgd',
                                        'lr'  : 0.001, },
                                       {'name': 'sgd',
                                        'lr'  : 0.01, },
                                       ],
                        'with_bn'   : [True, False]
                        }
  },
  # cifar10 all
  "cifar10_all" : {
    "epochs"         : 301,
    "sample_rate"    : 1,  # epoch^-1
    "sub_sample"     : [10, 30],  # epoch
    "sub_sample_rate": [6, 3, 1],
    "grids"          : {'dataset'   : ['cifar10', 'cifar100', 'mnist', 'svhn'],
                        'model_type': ['resnet10', 'vgg10', ],
                        'optimizer' : [{'name': 'sgd',
                                        'lr'  : 0.001, },
                                       {'name': 'sgd',
                                        'lr'  : 0.01, },
                                       {'name'       : 'sgd',
                                        'lr'         : 0.01,
                                        'decay_epoch': [150, ],
                                        'decay'      : [10, ], },
                                       {'name'       : 'sgd',
                                        'lr'         : 0.01,
                                        'decay_epoch': [50, ],
                                        'decay'      : [10, ], },
                                       ],
                        'with_bn'   : [True, False]
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
  # lr search
  "lr_search"   : {
    "epochs"         : 201,
    "sample_rate"    : 1,  # epoch^-1
    "sub_sample"     : [10, 30],  # epoch
    "sub_sample_rate": [3, 2, 1],
    'grids'          : {
      'dataset'   : ['cifar10'],  # , 'cifar100'
      'model_type': ['resnet10', 'vgg10'],  # 'vgg8', 'resnet8','vgg6', 'resnet6',
      'optimizer' : [{'name'       : 'sgd',
                      'lr'         : 0.01,
                      'decay_epoch': [150, ],
                      'decay'      : [10, ], },
                     {'name'       : 'sgd',
                      'lr'         : 0.01,
                      'decay_epoch': [50, ],
                      'decay'      : [10, ], },
                     # {'name': 'sgd',
                     #  'lr'  : 0.001,
                     #  },
                     # {'name': 'sgd',
                     #  'lr'  : 0.01,
                     #  },
                     # {'name': 'adam',
                     #  'lr'  : 0.001},
                     ]
    }
  },
}
