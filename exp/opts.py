from __future__ import division

import argparse, sys, json
import numpy as np
import os.path as osp
from math import log


def mkdir_p(path, delete=True):
    import subprocess, os
    if delete:
        subprocess.call(('rm -rf ' + path).split())
    if not os.path.exists(path):
        subprocess.call(('mkdir -p ' + path).split())


class Config(object):
    # shared across model
    root_path = osp.normpath(
        osp.join(osp.dirname(__file__), "..")
    )
    output_path = osp.join(root_path, 'output')
    tfevents_path = osp.join(root_path, 'tfevents')
    stream_verbose = True

    def __init__(self, epochs=100, batch_size=256, verbose=1, name=None, model_type='vgg11',
                 dataset_type='cifar10', debug=False, others=None):
        self.debug = debug
        self.model_type = model_type
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        if name is None:
            self.name = name = model_type + '_' + dataset_type
            if others is not None:
                for key, val in others.iteritems():
                    # if isinstance(val, bool): key, val =( key, '') if val else ('', '')
                    if isinstance(val, bool): key, val =( key, 'T') if val else (key, 'F')
                    name += '_' + str(key) + '_' + str(val)
                self.name = name

        self.model_tfevents_path = osp.join(Config.tfevents_path, name)
        self.model_output_path = osp.join(Config.output_path, name)
        self.clean_model_path()
        self.epochs = epochs
        self.verbose = verbose

    def copy(self, name='diff_name'):
        # todo
        new_config = Config()
        return new_config

    def clean_model_path(self):
        mkdir_p(self.model_tfevents_path)
        mkdir_p(self.model_output_path)


if __name__ == '__main__':
    config = Config(epochs=1, verbose=2, name='config', dataset_type='cifar10')
