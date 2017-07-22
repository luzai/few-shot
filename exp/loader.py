import time, glob
import pandas as pd
from log import logger
from opts import Config
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
import tensorflow as tf
from tensorflow.contrib.util import make_ndarray
import numpy as np


class Loader(object):
    def __init__(self, path):
        if 'events.out.tfevents' not in path:
            path = glob.glob(path + '/*')[0]
        self.path = path
        self.em = event_accumulator.EventAccumulator(
            size_guidance={event_accumulator.COMPRESSED_HISTOGRAMS: 1,
                           event_accumulator.IMAGES: 1,
                           event_accumulator.AUDIO: 1,
                           event_accumulator.SCALARS: 0,
                           event_accumulator.HISTOGRAMS: 1,
                           event_accumulator.TENSORS: 0},
            path=path)
        tic = time.time()
        self.em.Reload()
        logger.info('reload consume time {}'.format(time.time() - tic))
        self.scalars_names = self.em.Tags()['scalars']
        self.tensors_names = self.em.Tags()['tensors']


class ScalarLoader(Loader):
    def __init__(self, name=None, path=None):
        self.name = name
        super(ScalarLoader, self).__init__(path)

    def load_scalars(self):
        scalars = {}
        for scalar_name in self.scalars_names:
            scalars[scalar_name] = [e.value for e in self.em.Scalars(scalar_name)]
        scalars_df = pd.DataFrame.from_dict(scalars)
        scalars_df.sort_index(axis=0, inplace=True)
        scalars_df.sort_index(axis=1, inplace=True)
        return scalars_df


class TensorLoader(Loader):
    def __init__(self, name=None, path=None):
        self.name = name
        super(TensorLoader, self).__init__(path)

    def load_tensors(self):
        tensors = pd.DataFrame()
        for tensor_name in self.tensors_names:
            tensors_t = pd.Series(name=clean_name(tensor_name))
            tensors_t.index = tensors_t.index.astype('int64')
            for e in self.em.Tensors(tensor_name):
                now_step = e.step
                now_tensor = make_ndarray(e.tensor_proto)
                # logger.info('step {} tensors {} shape {}'.format(now_step, tensor_name, now_tensor.shape))
                if now_step not in tensors_t:
                    tensors_t[now_step] = now_tensor
                else:
                    tensors_t[now_step] = np.concatenate((now_tensor, tensors_t[now_step]), axis=0)
            tensors = pd.concat((tensors, tensors_t), axis=1)
        tensors.sort_index(axis=0, inplace=True)
        tensors.sort_index(axis=1, inplace=True)
        return tensors

    def head_tensors(self):
        for tensor_name in self.tensors_names:
            for e in self.em.Tensors(tensor_name):
                now_step = e.step
                now_tensor = make_ndarray(e.tensor_proto)
                logger.info('step {} tensors {} shape {}'.format(now_step, tensor_name, now_tensor.shape))


class ParamLoader(object):
    def __init__(self, name=None, path=None):
        self.name = name
        self.path_l = glob.glob(path + '/*')

    def load_param(self):
        tensors_l = []
        for path in self.path_l:
            tensors_l.append(TensorLoader(path=path).load_tensors())
        tensors = pd.concat(tensors_l)
        tensors.sort_index(axis=0, inplace=True)
        tensors.sort_index(axis=1, inplace=True)
        return tensors


class ActLoader(object):
    def __init__(self, name=None, path_l=None):
        self.name = name
        self.path_l = path_l

    def load_act(self):
        pass

def clean_name(name):
    import re
    name = re.findall('([a-zA-Z0-9/]+)(?::\d+)?', name)[0]
    name = re.findall('([a-zA-Z0-9/]+)(?:_\d+)?', name)[0]
    return name

def test_df(df):
    print df.index
    print df.columns


if __name__ == '__main__':
    tic = time.time()
    path = Config.root_path + '/tfevents/vgg11_cifar10/miscellany'
    scalars = ScalarLoader(path=path).load_scalars()
    logger.info('load scalars consmue {}'.format(time.time() - tic))
    print scalars

    # path_l = glob.glob(Config.root_path + '/tfevents/vgg11_cifar10/act/*/*/*')
    # acts = ActLoader(path_l=path_l).load_act()
    # test_df(acts)

    path = Config.root_path + '/tfevents/vgg11_cifar10/param'
    params = ParamLoader(path=path).load_param()
    test_df(params)
