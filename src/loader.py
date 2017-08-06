import time, glob, os, os.path as osp, re, pandas as pd
from logs import logger
from configs import Config
from tensorflow.tensorboard.backend.event_processing import event_accumulator
# from tensorflow.tensorboard.backend.event_processing import event_multiplexer
import tensorflow as tf
from tensorflow.contrib.util import make_ndarray
import numpy as np
from pathos.pools import ProcessPool as Pool
import utils
from utils import timer
from stats import Stat
import threading
from utils import clean_name

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
        scalars_df=pd.DataFrame()
        for scalar_name in self.scalars_names:
            for e in self.em.Scalars(scalar_name):
                iter = e.step
                val = e.value
                scalars_df.loc[iter,scalar_name] = val

        return scalars_df.sort_index().sort_index(axis=1)


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


def df_sort_index(tensors):
    tensors.sort_index(axis=0, inplace=True)
    tensors.sort_index(axis=1, inplace=True)
    return tensors




def test_df(df):
    print df.index
    print df.columns


def check_cache(path):
    path += '/cache'
    if osp.exists(path + '.pkl'):
        res = utils.unpickle(path)
        res.to_hdf(path.rstrip('.pkl') + '.h5', 'df', mode='w')
        return res
    elif osp.exists(path + '.h5'):
        res = pd.read_hdf(path + '.h5', 'df')
        return res
    else:
        return None


def cache(res, path, delete=False):
    utils.mkdir_p(path, delete=False)
    res.to_hdf(path + '/cache.h5', 'df', mode='w')
    if delete:
        path = path.rstrip('/cache.h5') + '/*'
        path_l = glob.glob(path)
        for path in path_l:
            if 'cache.' not in path:
                utils.rm(path)


class MultiLoader(object):
    def __init__(self, path=None, name=None):
        self.name = name
        self.path = path
        self.path_l = glob.glob(path + '/*')

    # @utils.timeit(info='parallel load takes')
    def parallel_load(self):
        res = check_cache(self.path)
        if res is not None:
            return res
        pool = Pool()  # ncpus=6
        tensors_l = pool.map(self._load, self.path_l)
        res = df_sort_index(pd.concat(tensors_l)) if tensors_l != [] else None
        cache(res, self.path)
        return res

    # @utils.timeit(info='series load takes')
    def seq_load(self):
        res = check_cache(self.path)
        if res is not None:
            return res
        tensors_l = []
        for path in self.path_l:
            logger.info('load from path {}'.format(path))
            tensors_l.append(self._load(path))
        res = df_sort_index(pd.concat(tensors_l)) if tensors_l != [] else None
        cache(res, self.path)
        return res

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


def select(df, pattern):
    poss_name = df.columns
    selected_name = set()
    pattern = re.compile(pattern)
    for _name in poss_name:
        judge = bool(pattern.match(_name))
        if judge: selected_name.add(_name)
    df = df.loc[:, (selected_name)]
    return df


class Loader(threading.Thread):
    def __init__(self, name=None, path=None, parallel=True, stat_only=False, cache=True, delete=False):  #
        threading.Thread.__init__(self)
        self.name = name
        self.path = path
        assert osp.exists(path), 'path should exits'
        self.parallel = parallel
        self.stat_only = stat_only

    def run(self):
        self.load(self.parallel, self.stat_only)

    def load(self, parallel=True, stat_only=False):
        timer = utils.Timer()
        timer.tic()
        path = self.path + '/miscellany'
        res = check_cache(path)
        if res is not None:
            self.scalars = res
        else:
            self.scalars = ScalarLoader(path=path).load_scalars()
            cache(select(self.scalars, "(?:val_loss|loss|val_acc|acc)"), path)
        logger.info('load scalars consume {}'.format(timer.toc()))

        if not stat_only:
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
        else:
            df = self.scalars
            self.scalars = select(df, "(?:val_loss|loss|val_acc|acc)")  # .columns

            path = self.path + '/act'
            res = check_cache(path)
            if res is not None:
                self.act = res
            else:
                self.act = select(df, "^obs.*?act.*")  # .columns
                cache(self.act, path)

            path = self.path + '/params'
            res = check_cache(path)
            if res is not None:
                self.params = res
            else:
                self.params = select(df, "^obs.*?(?:kernel|bias).*")
                cache(self.params, path)


if __name__ == '__main__':

    for path in glob.glob(Config.root_path + '/stat/*'):
        print path
        # try:
        loader = Loader(path=path)
        loader.load(stat_only=True, parallel=True)
        print loader.scalars
        # except Exception as inst:
        #     print  inst
        # from IPython import embed;embed()