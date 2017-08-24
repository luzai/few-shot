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


def copy_model(model):
  path = 'model.h5'
  keras.models.save_model(model, path)
  new_model = keras.models.load_model(path)
  assert new_model is not model
  return new_model


def get_layer_names(model):
  layer_names = [l.name for l in model.layers if
                 'input' not in l.name.lower() and
                 'maxpooling2d' not in l.name.lower() and
                 'add' not in l.name.lower() and
                 'dropout' not in l.name.lower() and
                 'concatenate' not in l.name.lower()]
  return layer_names


def evaluate(model, x=None, y=None, verbose=0):
  if x is None:
    dataset = Dataset('cifar10')
    x = dataset.x_test
    y = dataset.y_test
  res = model.evaluate(batch_size=256, x=x, y=y, verbose=verbose)
  return res[1]


# def


def ortho(tt):
  # print tt.shape
  angles = orthochnl(tt, single=False)
  # print angles.shape
  np.fill_diagonal(angles, np.nan)
  
  return np.nanmax(angles, axis=1).sum() / tt.shape[0]


def series2tensor(t, flatten=False):
  if flatten:
    return np.array(t.values.tolist()).squeeze()
  else:
    return np.array(t.values).squeeze()


def orthochnl(tensor, single=True):
  tensor = tensor.reshape(-1, tensor.shape[-1])
  shape1, shape2 = tensor.shape
  tensor = tensor.T
  tensor = tensor / np.linalg.norm(tensor, axis=1)[:, np.newaxis]
  angles = np.dot(tensor, tensor.T)
  logger.debug('angles matrix is {}'.format(angles.shape))
  if single:
    np.fill_diagonal(angles, np.nan)
    return np.nanmean(angles)
  else:
    
    return angles


def orthosmpl(tensor, single=True):
  tensor = tensor.reshape(tensor.shape[0], -1)
  shape1, shape2 = tensor.shape
  # print tensor.shape
  tensor = tensor / np.linalg.norm(tensor, axis=1)[:, np.newaxis]
  angles = np.dot(tensor, tensor.T)
  logger.debug('angles matrix is {}'.format(angles.shape))
  if single:
    np.fill_diagonal(angles, np.nan)
    return np.nanmean(angles)
  else:
    return angles


def orthogonalize(weights):
  flat_shape = shape = weights.shape
  u, _, v = np.linalg.svd(weights, full_matrices=False)
  # pick the one with the correct shape
  q = u if u.shape == flat_shape else v
  q = q.reshape(shape)
  u_new = q
  return u_new


# def perm_iter(U):
#   import itertools
#   for ind, u in enumerate(itertools.permutations(U.transpose())):
#     u_new = np.array(u).transpose()
#     yield u_new


def perm(weights):
  np.random.seed(np.uint8(time.time() * 100))
  return np.random.permutation(weights.transpose()).transpose()


def get_one_class_data(c, dataset=None, limit=False):
  if dataset is None:
    dataset = Dataset('cifar10')
  y_ori = np.where(dataset.y_test)[1]
  pos = np.where(y_ori == c)[0]
  x, y = dataset.x_test[pos], dataset.y_test[pos]
  if limit:
    x, y = x[:100], y[:100]
  return x, y


def migrate_to(model1, model2, copy=False):
  if isinstance(model1, keras.models.Model):
    model1 = copy_model(model1)
    weight = get_last_weight(model1)
  else:
    weight = model1
  if copy:
    model2 = copy_model(model2)
  
  model2.get_layer(get_last_name(model2)).set_weights([weight])
  return model2


def get_last_name(model):
  names = get_layer_names(model)
  for name in names[::-1]:
    if 'softmax' in name.lower(): continue
    if 'dense' in name.lower(): break
  return name


def get_last_weight(model):
  names = get_layer_names(model)
  for name in names[::-1]:
    if 'softmax' in name.lower(): continue
    if 'dense' in name.lower(): break
  return model.get_layer(name).get_weights()[0]


def exchange_col(weight, ind1, ind2):
  weight2 = weight.copy()
  weight2[:, ind1], weight2[:, ind2] = weight[:, ind2], weight[:, ind1]
  return weight2


def custom_sort(tensor, y):
  comb = zip(tensor, y)
  res = sorted(comb, key=lambda x: x[1])
  t = np.array(res)
  t = t[:, 0]
  tt = [tensor_.tolist() for tensor_ in t]
  return np.array(tt)


def gen_fake():
  fake = np.arange(20).reshape(5, 4)
  return fake


def calc_margin(inp, out):
  inp = inp.squeeze()
  out = out.squeeze()
  dataset = Dataset('cifar10')
  y_ori = np.where(dataset.y_test_ref)[1]
  x_norm = np.linalg.norm(inp, axis=1)
  
  out_ori = out[np.arange(out.shape[0]), y_ori]
  out_t = out.copy()
  out_t[np.arange(out.shape[0]), y_ori] = out_t.min(axis=1) - 1
  
  out_max = np.nanmax(out_t, axis=1)
  res = (out_ori - out_max) / x_norm
  return res


def hasnan(t):
  return np.isnan(t).any()


if __name__ == '__main__':
  tensor = np.random.rand(272, 5)
