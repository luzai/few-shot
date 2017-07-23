import time, glob
import pandas as pd
from log import logger
from opts import Config
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
import tensorflow as tf
from tensorflow.contrib.util import make_ndarray
import numpy as np
from pathos.pools import ProcessPool as Pool
from utils import timer


class Loader(object):
    def __init__(self, name, path):
        self.name = name
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
        self.reload()

    def reload(self):
        tic = time.time()
        self.em.Reload()
        # logger.info('reload consume time {}'.format(time.time() - tic))
        self.scalars_names = self.em.Tags()['scalars']
        self.tensors_names = self.em.Tags()['tensors']


class ScalarLoader(Loader):
    def __init__(self, name=None, path=None):
        super(ScalarLoader, self).__init__(name, path)

    def load_scalars(self, reload=False):
        if reload:
            self.reload()
        scalars = {}
        for scalar_name in self.scalars_names:
            scalars[scalar_name] = [e.value for e in self.em.Scalars(scalar_name)]
        scalars_df = pd.DataFrame.from_dict(scalars)
        scalars_df.sort_index(axis=0, inplace=True)
        scalars_df.sort_index(axis=1, inplace=True)
        return scalars_df


class TensorLoader(Loader):
    def __init__(self, name=None, path=None):
        super(TensorLoader, self).__init__(name, path)

    def load_tensors(self, reload=False):
        if reload:
            self.reload()
        tensors = pd.DataFrame()
        for tensor_name in self.tensors_names:
            tensors_t = pd.Series(name=clean_name(tensor_name), index=np.array([], dtype='int64'))
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


class Stat(object):
    def __init__(self, stat='all'):
        if stat == 'all':
            self.stat = dict(Stat.__dict__)
            for key in self.stat.keys():
                if '_' in key:
                    del self.stat[key]

    def min(self, tensor):
        return tensor.min()

    def max(self, tensor):
        return tensor.max()

    def mean(self, tensor):
        return tensor.mean()

    def std(self, tensor):
        return tensor.std()


def df_sort_index(tensors):
    tensors.sort_index(axis=0, inplace=True)
    tensors.sort_index(axis=1, inplace=True)
    return tensors


class ParamLoader(Stat):
    def __init__(self, path=None, name=None):
        self.name = name
        self.path_l = glob.glob(path + '/*')
        super(ParamLoader, self).__init__()

    def _load(self, path):
        _tensor = TensorLoader(path=path).load_tensors()
        tensor = pd.DataFrame()
        for name, series in _tensor.iteritems():
            for ind, val in series.iteritems():
                for stat_name, stat_func in self.stat.iteritems():
                    tensor.loc[ind, name + '/' + stat_name] = stat_func(self, val)
        return tensor

    def parallel_load(self):
        pool = Pool()
        tensos_l = pool.map(self._load, self.path_l)
        return df_sort_index(pd.concat(tensos_l))

    def seq_load(self):
        tensors_l = []
        for path in self.path_l:
            tensors_l.append(self._load(path))
        return df_sort_index(pd.concat(tensors_l))


class ActLoader(Stat):
    def __init__(self, path=None, name=None):
        self.name = name
        self.path_l = glob.glob(path + '/*')
        super(ActLoader, self).__init__()

    def _load(self, path):
        tensor_l = []
        for path_ in glob.glob(path + '/*'):
            tensor_l.append(TensorLoader(path=path_).load_tensors())
        acts_t = pd.concat(tensor_l)
        acts_t.reset_index(inplace=True)
        tensors = {}
        for epoch, group in acts_t.groupby('index'):
            tensors_t = {}
            for name, series in group.iteritems():
                if name == 'index': continue
                val_l = []
                for ind, val in series.iteritems():
                    val_l.append(val)
                res = np.concatenate(val_l)
                tensors_t[name] = res
            tensors[epoch] = tensors_t
        tensors = pd.DataFrame(tensors)
        # timer.toc()
        tensors = tensors.transpose()
        tensors.sort_index(axis=0, inplace=True)
        tensors.sort_index(axis=1, inplace=True)

        tensor = pd.DataFrame()
        for name, series in tensors.iteritems():
            for ind, val in series.iteritems():
                for stat_name, stat_func in self.stat.iteritems():
                    tensor.loc[ind, name + '/' + stat_name] = stat_func(self, val)
        return tensor

    def parallel_load(self):
        pool = Pool()
        tensos_l = pool.map(self._load, self.path_l)
        return df_sort_index(pd.concat(tensos_l))

    def seq_load(self):
        tensors_l = []
        for path in self.path_l:
            tensors_l.append(self._load(path))
        return df_sort_index(pd.concat(tensors_l))


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
    # print scalars
    timer.tic()

    path = Config.root_path + '/tfevents/vgg11_cifar10/act'
    acts = ActLoader(path=path).parallel_load()
    timer.toc()
    test_df(acts)

    path = Config.root_path + '/tfevents/vgg11_cifar10/param'
    params = ParamLoader(path=path).parallel_load()
    timer.toc()
    test_df(params)
