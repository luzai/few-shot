from __future__ import division

import argparse, sys, json
import numpy as np
import os.path as osp
import utils


class Config(object):
  # shared across model
  root_path = utils.root_path
  output_path = osp.join(root_path, 'output')
  tfevents_path = osp.join(root_path, 'tfevents')
  
  def __init__(self, epochs=100, batch_size=256, verbose=1, name=None, model_type='vgg10',
               dataset_type='cifar10', debug=False, others=None, clean=False, clean_after=False):
    self.debug = debug
    self.model_type = model_type
    self.batch_size = batch_size
    self.dataset_type = dataset_type
    self.others = others
    self.clean_after = clean_after
    if name is None:
      self.name = name = model_type + '_' + dataset_type
      if others is not None:
        for key, val in others.iteritems():
          if (isinstance(val, float) or
                isinstance(val, int)) \
              and key == 'lr':
            key, val = key, '{:.2e}'.format(val)
          name += '_' + str(key) + '_' + str(val)
    self.name = name
    
    self.model_tfevents_path = osp.join(Config.tfevents_path, name)
    self.model_output_path = osp.join(Config.output_path, name)
    
    self.clean_model_path(clean)
    self.epochs = epochs
    self.verbose = verbose
    self.to_pkl()
  
  def to_dict(self):
    d = {'model_type'  : self.model_type,
         'dataset_type': self.dataset_type}
    d = d.copy()
    if self.others is not None:  d.update(self.others)
    if 'lr' in d: d['lr'] = '{:.2e}'.format(d['lr'])
    return d
  
  def clean_model_path(self, clean):
    utils.mkdir_p(self.model_tfevents_path, delete=clean)
    utils.mkdir_p(self.model_output_path, delete=clean)
  
  def to_pkl(self):
    utils.pickle(self, self.model_tfevents_path + '/config.pkl')


if __name__ == '__main__':
  config = Config(epochs=1, verbose=2, dataset_type='cifar10')
  print config.to_dict()
  config.to_pkl()
