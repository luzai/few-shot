import os, csv, time, cPickle, \
  random, os.path as osp, \
  subprocess, json, matplotlib, \
  numpy as np, GPUtil, pandas as pd, \
  glob, re, keras
import keras.backend as K
from logs import logger
from datasets import Dataset


def rand_weight_like(weight):
  assert K.image_data_format() == "channels_last", "support channels last, but you are {}".format(
      K.image_data_format())
  kw, kh, num_channel, filters = weight.shape
  kvar = K.truncated_normal((kw, kh, num_channel, filters), 0, 0.05)
  w = K.eval(kvar)
  b = np.zeros((filters,))
  return w, b


def copy_weight(before_model, after_model):
  layer_names = [l.name for l in before_model.layers if
                 'input' not in l.name.lower() and
                 'maxpooling2d' not in l.name.lower() and
                 'add' not in l.name.lower() and
                 'dropout' not in l.name.lower() and
                 'concatenate' not in l.name.lower()]
  for name in layer_names:
    weights = before_model.get_layer(name=name).get_weights()
    try:
      after_model.get_layer(name=name).set_weights(weights)
    except Exception as inst:
      logger.warning("ignore copy layer {} from model {} to model {} because {}".format(name,
                                                                                        before_model.config.name,
                                                                                        after_model.config.name,
                                                                                        inst))


def copy_model(path):
  new_model = keras.models.load_model(path)
  # keras.models.save_model(new_model, path)
  return new_model


def get_layer_names(model):
  layer_names = [l.name for l in model.layers if
                 'input' not in l.name.lower() and
                 'maxpooling2d' not in l.name.lower() and
                 'add' not in l.name.lower() and
                 'dropout' not in l.name.lower() and
                 'concatenate' not in l.name.lower()]
  return layer_names


def evaluate(model, x=None, y=None):
  if x is None:
    dataset = Dataset('cifar10')
    x = dataset.x_test
    y = dataset.y_test
  res = model.evaluate(batch_size=256, x=x, y=y, verbose=0)
  return res[1]


def orthochnl(tensor, name=None, iter=None, axis=-1):
  tensor = tensor.reshape(-1, tensor.shape[axis])
  shape1, shape2 = tensor.shape
  tensor = tensor.T
  tensor = tensor / np.linalg.norm(tensor, axis=1)[:, np.newaxis]
  angles = np.dot(tensor, tensor.T)
  logger.debug('angles matrix is {}'.format(angles.shape))
  np.fill_diagonal(angles, np.nan)
  return np.nanmean(angles)


def get_ortho(weights):
  flat_shape = shape = weights.shape
  u, _, v = np.linalg.svd(weights, full_matrices=False)
  # pick the one with the correct shape
  q = u if u.shape == flat_shape else v
  q = q.reshape(shape)
  u_new = q
  return u_new


def get_perm_iter(U):
  import itertools
  for ind, u in enumerate(itertools.permutations(U.transpose())):
    u_new = np.array(u).transpose()
    yield u_new


def get_perm(weights):
  np.random.seed(np.uint8(time.time() * 100))
  return np.random.permutation(weights.transpose()).transpose()


def get_data(c, dataset):
  y_ori = np.where(dataset.y_test)[1]
  pos = np.where(y_ori == c)[0]
  return dataset.x_test[pos], dataset.y_test[pos]
