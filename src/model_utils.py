import os, csv, time, cPickle, \
  random, os.path as osp, \
  subprocess, json, matplotlib, \
  numpy as np, GPUtil, pandas as pd, \
  glob, re,keras
import keras.backend as K
from logs import logger

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython import embed
from IPython.display import display, HTML, SVG


def rand_weight_like(weight):
  assert K.image_data_format() == "channels_last", "support channels last, but you are {}".format(
      K.image_data_format())
  kw, kh, num_channel, filters = weight.shape
  kvar = K.truncated_normal((kw, kh, num_channel, filters), 0, 0.05)
  w = K.eval(kvar)
  b = np.zeros((filters,))
  return w, b

def copy_weight(self, before_model, after_model):
  _before_model = self.my_model2model(model=before_model)
  _after_model = self.my_model2model(model=after_model)
  
  layer_names = [l.name for l in _before_model.layers if
                 'input' not in l.name.lower() and
                 'maxpooling2d' not in l.name.lower() and
                 'add' not in l.name.lower() and
                 'dropout' not in l.name.lower() and
                 'concatenate' not in l.name.lower()]
  for name in layer_names:
    weights = _before_model.get_layer(name=name).get_weights()
    try:
      _after_model.get_layer(name=name).set_weights(weights)
    except Exception as inst:
      logger.warning("ignore copy layer {} from model {} to model {} because {}".format(name,
                                                                                        before_model.config.name,
                                                                                        after_model.config.name,
                                                                                        inst))

# def copy_model( model, config):
#     from keras.utils.generic_utils import get_custom_objects
#
#     new_model = MyModel(config, model.graph.copy(), keras.models.load_model(model.config.model_path))
#     keras.models.save_model(new_model.model, new_model.config.model_path)
#     return new_model
