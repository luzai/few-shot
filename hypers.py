import numpy as np


def list2str(ls):
    s = ''
    for l in ls:
        s += ' '
        s += str(l)
    return s


def str2list(s):
    ls = s.split()
    return [float(l) for l in ls]


hyper = {
    "gpu": [0, 1, 2, 3],
    "use": "cifar",
    "logger_level": "info",
    "dbg": False,
    "priority": 19,  # 11
    "log_stat": True,
    # we need 2 mode : log all stat
    # stat and tensor for only Layer
    "log_tensor": False,
    "last_only": True,
    "curve_only": True,

    "cifar": {
        "epochs": 81,
        "sample_rate": 1.,  # epoch^-1
        "sub_sample": [5, 20, ],  # epoch
        "sub_sample_rate": [20, 2, .5],
        "grids": {'dataset': ['cifar100', ],
                  'model_type': ['resnet101'],  # , 'vgg10', 'resnet10',
                  'optimizer': [
                      # {'name'       : 'sgd',
                      #  'lr'         : 0.01,
                      #  'decay_epoch': [49, ],
                      #  'decay'      : [10, ], },
                      # {'name'       : 'sgd',
                      #  'lr'         : 0.01,
                      #  'decay_epoch': [25, ],
                      #  'decay'      : [10, ], },
                      {'name': 'sgd',
                       'lr': 0.001},
                  ],
                  # 'hiddens': [2, ],
                  'loss': ['softmax', ],
                  # 'classes': [10, ],
                  'ortho_l2': [
                      list2str([0.0, 1e-4]),
                      # list2str([1e-4, 0.0]),
                      # list2str([1e-4, 1e-4])
                  ],
                  }
    },
    # # mnist ok
    # "mnist": {
    #     "epochs": 5,
    #     "sample_rate": 10,
    #     "sub_sample": [1, 2],
    #     "sub_sample_rate": [5, 3, 1],
    #     "grids": {'dataset': ['mnist', ],  # 'cifar10'],
    #               'model_type': ['resnet10'],  # 'vgg16',  'vgg19'
    #               'optimizer': [{'name': 'sgd',
    #                              'lr': 0.001,
    #                              },
    #                             {'name': 'sgd',
    #                              'lr': 0.01,
    #                              },
    #                             ],
    #               }
    # },
    # 'vgg': {
    #     "epochs": 61,
    #     "grids": {'dataset': ['cifar10', ],  # 'cifar10'],
    #               'model_type': ['vgg101', 'vgg102', 'vgg103', 'resnet10'],  # 'vgg16', , 'vgg19'
    #               'optimizer': [{'name': 'sgd',
    #                              'lr': 0.01, },
    #                             ],
    #               'hiddens': [5, 20, 512]
    #               }
    # },

    "win_size": 11,
    # "use_bias"    : False,  # todo

}
