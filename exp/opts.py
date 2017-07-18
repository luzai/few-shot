from __future__ import division

import argparse, sys, json
import numpy as np
import os.path as osp
from math import log


class Config(object):
    root_dir =root_path = osp.normpath(
        osp.join(osp.dirname(__file__), "..")
    )
    logger_path = output_path = osp.join(root_dir, 'output')
    tf_path = osp.join(root_dir, 'tf_log')
    stream_verbose = True

    def __init__(self, epochs=100, verbose=1, limit_data=False, name='default_name', evoluation_time=1, clean=True,
                 dataset_type='cifar10', max_pooling_cnt=0, debug=False):
        # for all model:
        self.dataset_type = dataset_type
        self.limit_data = limit_data
        if dataset_type == 'cifar10' or dataset_type == 'svhn' or dataset_type == 'cifar100':
            self.input_shape = (32, 32, 3)
        else:
            self.input_shape = (28, 28, 1)
        self.nb_class = 10
        self.dataset = None
        if limit_data:
            self.load_data(9999, type=self.dataset_type)
        else:
            self.load_data(1, type=self.dataset_type)

        # for ga:
        self.evoluation_time = evoluation_time

        # for single model
        self.set_name(name, clean=clean)
        self.batch_size = 256
        self.epochs = epochs
        self.verbose = verbose
        self.lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10,
                                            min_lr=0.5e-7)
        self.early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5)
        self.csv_logger = None
        self.set_logger_path(self.name + '.csv')
        self.debug = debug
        self.max_pooling_limit = int(log(min(self.input_shape[0], self.input_shape[1]), 2)) - 2
        self.max_pooling_cnt = max_pooling_cnt

        self.model_max_conv_width = 1024
        self.model_min_conv_width = 128
        self.model_max_depth = 20
        self.kernel_regularizer_l2 = 0.0

    def set_logger_path(self, name):
        self.csv_logger = CSVLogger(osp.join(self.output_path, name))

    def to_json(self):
        d = dict(name=self.name,
                 epochs=self.epochs,
                 verbose=self.verbose)
        with open(osp.join(self.output_path, 'config.json')) as f:
            json.dumps(f, d)
        return

    def copy(self, name='diff_name'):
        new_config = MyConfig(self.epochs, self.verbose, limit_data=self.limit_data, name=name,
                              evoluation_time=self.evoluation_time, dataset_type=self.dataset_type,
                              max_pooling_cnt=self.max_pooling_cnt)
        return new_config

    def set_name(self, name, clean=True):
        self.name = name
        self.tf_log_path = osp.join(root_dir, 'output/tf_tmp/', name)
        self.output_path = osp.join(root_dir, 'output/', name)
        self.model_path = osp.join(root_dir, 'output/', name, name + '.h5')
        if clean:
            Utils.mkdir_p(self.tf_log_path)
            Utils.mkdir_p(self.output_path)


if __name__ == '__main__':
    config = MyConfig(epochs=1, verbose=2, limit_data=False, name='ga', evoluation_time=3, dataset_type='mnist')
