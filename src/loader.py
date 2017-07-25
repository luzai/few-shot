import time, glob, os, os.path as osp
import pandas as pd
import utils
from log import logger
from opts import Config
from tensorflow.tensorboard.backend.event_processing import event_accumulator
# from tensorflow.tensorboard.backend.event_processing import event_multiplexer
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
        scalars_df = pd.DataFrame.from_dict(dict([(k, pd.Series(v)) for k, v in scalars.iteritems()]))
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

    # todo more and specified for weight
    def min(self, tensor):
        return tensor.min()

    def max(self, tensor):
        return tensor.max()

    def mean(self, tensor):
        return tensor.mean()

    def median(self, tensor):
        return np.median(tensor)

    def std(self, tensor):
        return tensor.std()

    def iqr(self, tensor):
        return np.subtract.reduce(np.percentile(tensor, [75, 25]))

    def pos_mean(self, tensor):
        return tensor[tensor > 0].mean()

    def neg_mean(self, tensor):
        return tensor[tensor < 0].mean()

    def pos_neg_rat(self, tensor):
        return float(tensor[tensor > 0].shape[0]) / float(tensor[tensor < 0].shape[0])


def df_sort_index(tensors):
    tensors.sort_index(axis=0, inplace=True)
    tensors.sort_index(axis=1, inplace=True)
    return tensors


def clean_name(name):
    import re
    name = re.findall('([a-zA-Z0-9/]+)(?::\d+)?', name)[0]
    name = re.findall('([a-zA-Z0-9/]+)(?:_\d+)?', name)[0]
    return name


def test_df(df):
    print df.index
    print df.columns


@utils.optional_arg_decorator
def check_cache(fn, cache=True, delete=True):
    def wrapped_fn(*args, **kwargs):
        if kwargs != {}:
            path = kwargs.get('self').path + '/cache.pkl'
        else:
            path = args[0].path + '/cache.pkl'
        if cache and not osp.exists(path):
            res = fn(*args, **kwargs)
            utils.pickle(res, path)
        elif cache and osp.exists(path):
            res = utils.unpickle(path)
        else:
            res = fn(*args, **kwargs)
        if delete:
            path = path.rstrip('/cache.pkl') + '/*'
            path_l = glob.glob(path)
            for path in path_l:
                if 'cache.pkl' not in path:
                    utils.rm(path)
        return res

    return wrapped_fn


class MultiLoader(object):
    def __init__(self, path=None, name=None):
        self.name = name
        self.path = path
        self.path_l = glob.glob(path + '/*')

    @check_cache
    # @utils.timeit(info='parallel load takes')
    def parallel_load(self):
        pool = Pool()  # ncpus=6
        tensors_l = pool.map(self._load, self.path_l)
        # pool.close()
        return df_sort_index(pd.concat(tensors_l)) if tensors_l != [] else None

    @check_cache
    # @utils.timeit(info='series load takes')
    def seq_load(self):
        tensors_l = []
        for path in self.path_l:
            tensors_l.append(self._load(path))
        return df_sort_index(pd.concat(tensors_l)) if tensors_l != [] else None

    def _load(self, path):
        pass


class ParamLoader(MultiLoader):
    def __init__(self, path=None, name=None):
        super(ParamLoader, self).__init__(path=path, name=name)

    def _load(self, path):
        stat = Stat()
        _tensor = TensorLoader(path=path).load_tensors()
        tensor = pd.DataFrame()
        for name, series in _tensor.iteritems():
            for ind, val in series.iteritems():
                for stat_name, stat_func in stat.stat.iteritems():
                    tensor.loc[ind, name + '/' + stat_name] = stat_func(stat, val)
        return tensor


class ActLoader(MultiLoader):
    def __init__(self, path=None, name=None):
        super(ActLoader, self).__init__(path=path, name=name)

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
        stat = Stat()
        tensor = pd.DataFrame()
        for name, series in tensors.iteritems():
            for ind, val in series.iteritems():
                for stat_name, stat_func in stat.stat.iteritems():
                    tensor.loc[ind, name + '/' + stat_name] = stat_func(stat, val)
        del tensors
        import gc
        gc.collect()
        return tensor


class Loader(object):
    def __init__(self, name=None, path=None):  # , cache=True, delete=False
        self.name = name
        self.path = path
        assert osp.exists(path), 'path should exits'

    def load(self, parallel=True):
        timer = utils.Timer()
        timer.tic()
        path = self.path + '/miscellany'
        self.scalars = ScalarLoader(path=path).load_scalars()
        path = self.path + '/act'
        if parallel:
            self.act = ActLoader(path=path).parallel_load()
        else:
            self.act = ActLoader(path=path).seq_load()
        logger.info('load act consume {}'.format(timer.toc()))
        path = self.path + '/param'
        if parallel:
            self.params = ParamLoader(path=path).parallel_load()
        else:
            self.params = ParamLoader(path=path).seq_load()
        logger.info('load param consume {}'.format(timer.toc()))


if __name__ == '__main__':
    path = '/home/wangxinglu/prj/Perf_Pred/tfevents/vgg11_cifar10_limit_val_T_lr_1'
    loader = Loader(path=path).load()
    # for path in glob.glob(Config.root_path + '/bak/*'):
    #     print path
    #     # try:
    #     loader = Loader(path=path).load()
    #     # except Exception as inst:
    #     #     print  inst
