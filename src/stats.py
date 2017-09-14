from __future__ import division
import numpy as np, Queue, pandas as pd
from logs import logger
import utils
import math

NOKEY = 200

NAN = float('nan')


def get_name2fn(class_name):
    return {key: val
            for key, val in class_name.__dict__.iteritems()
            if '_' not in key}


def thresh_proportion(arr, thresh):
    lower = float(arr[arr < thresh].shape[0])
    greater = float(arr[arr > thresh].shape[0])

    # thresh = max(thresh, arr.min())
    # thresh = min(thresh, arr.max())
    # hist, _ = np.histogram(arr, [arr.min(), thresh, arr.max() + np.finfo(float).eps])
    # hist = hist / hist.astype(float).sum()
    # assert np.allclose(hist.sum(), [1.]), 'histogram sum close to 1.'
    return lower / (lower + greater), greater / (lower + greater)


# todo piandu fengdu

class Stat(object):
    def __init__(self, max_win_size, log_pnt):
        self.common_interval = common_interval = np.diff(log_pnt.where(log_pnt == 3).dropna().index).min()
        self.log_pnt = log_pnt
        self.stdtime_inst = OnlineStd(common_interval)
        self.diff_inst = Diff(common_interval)

        self.window = windows = Windows(max_win_size=max_win_size)
        self.totvar_inst = TotVar(windows)

        self.stat = get_name2fn(Stat)

    # def diff(self, tensor, name=None, iter=None):
    #   if iter in self.log_pnt and self.log_pnt.loc[iter] == 3:
    #     return self.diff_inst.diff(tensor, iter, name)



    def stdtime(self, tensor, name=None, iter=None, how='mean'):
        if iter in self.log_pnt and self.log_pnt.loc[iter] == 3 or how == 'tensor':
            return self.stdtime_inst.online_std(tensor, iter, name, how=how)

    def min(self, tensor, **kwargs):
        return tensor.min()

    def max(self, tensor, **kwargs):
        return tensor.max()

    def mean(self, tensor, **kwargs):
        return tensor.mean()

    def median(self, tensor, **kwargs):
        return np.median(tensor)

    def std(self, tensor, **kwargs):
        return tensor.std()

    def iqr(self, tensor, **kwargs):
        return np.subtract.reduce(np.percentile(tensor, [75, 25]))

    # def posmean(self, tensor, **kwargs):
    #     return tensor[tensor > 0].mean()

    # def negmean(self, tensor, **kwargs):
    #     # todo do not observe softmax
    #     return tensor[tensor < 0].mean()

    # def posproportion(self, tensor, **kwargs):
    #     # in fact we use non-negative proportion
    #     pos_len = float(tensor[tensor >= 0].shape[0])
    #     neg_len = float(tensor[tensor < 0.].shape[0])
    #     res = pos_len / (pos_len + neg_len)
    #     _, res2 = thresh_proportion(tensor, 0.)
    #     if not np.allclose(np.array([res]), np.array(res2)):
    #         logger.error('different method calc pso_proportion should close {} {}'.format(res, res2))
    #     return res

    # def norm(self, tensor, **kwargs):
    #   return np.linalg.norm(tensor) / tensor.size

    def magnitude(self, tensor, **kwargs):
        return np.abs(tensor).mean()

    def totvar(self, tensor, name, iter, win_size):
        _iter, _val = self.totvar_inst.tot_var(tensor, iter, name, win_size, 'save')

        if self.log_pnt[iter] >= 1 \
                and np.in1d(np.arange(iter - win_size + 1, iter + 1, step=1), self.log_pnt.index).all():
            _iter, _val = self.totvar_inst.tot_var(tensor, iter, name, win_size, 'load')
            if _iter != iter - win_size // 2:
                logger.error('should log to right iter! {} {}'.format(_iter, iter - win_size // 2))

        return _iter, _val

    def calc_all(self, tensor, name, iter):
        calc_res = pd.DataFrame()
        level1 = ['totvar', 'ptrate']
        level2 = ['diff', 'stdtime']
        level3 = [key for key in self.stat.keys() if key not in level1 and key not in level2]
        if iter not in self.log_pnt:
            return calc_res

        @utils.timeit('calc level1 ')
        def calc_level1():
            logger.debug('level1 calc tensor shape {} name {} iter {}'.format(tensor.shape, name, iter))
            fn_name = 'totvar'
            if fn_name not in self.stat: return
            for win_size in [utils.get_config('win_size')]:
                _name = name + '/' + fn_name + '_win_size_' + str(win_size)
                _iter, _val = self.totvar(name=name, iter=iter, tensor=tensor, win_size=win_size)
                if math.isnan(_iter): continue
                calc_res.loc[_iter, _name] = _val

            fn_name = 'ptrate'
            if fn_name not in self.stat: return
            for thresh in ['mean']:  # .2, .6,
                for win_size in [utils.get_config('win_size')]:  # 31 , 21
                    _name = name + '/' + fn_name + '_win_size_' + str(win_size) + '_thresh_' + str(thresh)
                    _iter, _val = self.ptrate(name=name, iter=iter, tensor=tensor, win_size=win_size, thresh=thresh)
                    if math.isnan(_iter): continue
                    calc_res.loc[_iter, _name] = _val

        @utils.timeit('calc level2 or 3 ')
        def calc_level23(level):
            logger.debug('level23 calc tensor shape {} name {} iter {}'.format(tensor.shape, name, iter))
            for fn_name, fn in self.stat.iteritems():
                if fn_name not in level: continue
                _name = name + '/' + fn_name
                _val = fn(self, tensor, name=name, iter=iter)
                calc_res.loc[iter, _name] = _val

        if self.log_pnt[iter] == 3:
            calc_level1(), calc_level23(level2), calc_level23(level3)
        elif self.log_pnt[iter] == 2:
            calc_level23(level3), calc_level1()
        else:
            calc_level1()
        assert isinstance(calc_res, pd.DataFrame), 'should be dataframe'
        return calc_res


class KernelStat(Stat):
    def __init__(self, max_win_size, log_pnt):
        super(KernelStat, self).__init__(max_win_size=max_win_size, log_pnt=log_pnt)
        _stat = dict(KernelStat.__dict__)
        for key in _stat.keys():
            if '_' in key:
                del _stat[key]
        self.stat = utils.dict_concat([self.stat, _stat])
        self.totvar_inst = TotVar(self.window)

    def updateratio(self, tensor, name=None, iter=None):
        if iter in self.log_pnt and self.log_pnt.loc[iter] == 3:
            return self.diff_inst.diff(tensor, iter, name) / np.mean(np.abs(tensor))

    def sparsity(self, tensor, **kwargs):
        tensor = tensor.flatten()
        mean = tensor.mean()  # todo median or mean
        thresh = mean / 10.
        return float((tensor[tensor < thresh]).shape[0]) / float(tensor.shape[0])
        # for bias we usually set lr mult=0.1? --> bias is not sparse

    @utils.timeit('kernel ortho consume')
    def orthoabs(self, tensor, name=None, iter=None, axis=-1):
        tensor = tensor.reshape(-1, tensor.shape[axis])
        shape1, shape2 = tensor.shape
        tensor = tensor.T
        tensor = tensor / np.linalg.norm(tensor, axis=1)[:, np.newaxis]
        angles = np.abs(np.dot(tensor, tensor.T))
        logger.debug('angles matrix is {}'.format(angles.shape))
        np.fill_diagonal(angles, np.nan)
        return np.nanmean(angles)

    def ortho(self, tensor, name=None, iter=None, axis=-1):
        tensor = tensor.reshape(-1, tensor.shape[axis])
        shape1, shape2 = tensor.shape
        tensor = tensor.T
        tensor = tensor / np.linalg.norm(tensor, axis=1)[:, np.newaxis]
        angles = np.dot(tensor, tensor.T)
        logger.debug('angles matrix is {}'.format(angles.shape))
        np.fill_diagonal(angles, np.nan)
        return np.nanmean(angles)


class BiasStat(Stat):
    def __init__(self, max_win_size, log_pnt):
        super(BiasStat, self).__init__(max_win_size=max_win_size, log_pnt=log_pnt)
        _stat = dict(BiasStat.__dict__)
        for key in _stat.keys():
            if '_' in key:
                del _stat[key]
        self.stat = utils.dict_concat([self.stat, _stat])
        self.totvar_inst = TotVar(self.window)


class ActStat(Stat):
    def __init__(self, max_win_size, log_pnt):
        super(ActStat, self).__init__(max_win_size=max_win_size, log_pnt=log_pnt)
        _stat = dict(ActStat.__dict__)
        for key in _stat.keys():
            if '_' in key:
                del _stat[key]
        self.stat = utils.dict_concat([self.stat, _stat])
        self.ptrate_inst = PTRate(self.window)

    def sparsity(self, tensor, **kwargs):
        tensor = tensor.flatten()
        mean = tensor.mean()  # todo median or mean
        thresh = mean / 10.
        return float((tensor[tensor < thresh]).shape[0]) / float(tensor.shape[0])
        # for bias we usually set lr mult=0.1? --> bias is not sparse

    @utils.timeit('act ortho comsume ')
    def orthochnlabs(self, tensor, name=None, iter=None, axis=-1):
        tensor = tensor.reshape(-1, tensor.shape[axis])
        shape1, shape2 = tensor.shape
        tensor = tensor.T
        tensor = tensor / np.linalg.norm(tensor, axis=1)[:, np.newaxis]
        angles = np.abs(np.dot(tensor, tensor.T))
        logger.debug('angles matrix is {}'.format(angles.shape))
        np.fill_diagonal(angles, np.nan)
        return np.nanmean(angles)

    def orthochnl(self, tensor, name=None, iter=None, axis=-1):
        tensor = tensor.reshape(-1, tensor.shape[axis])
        shape1, shape2 = tensor.shape
        tensor = tensor.T
        tensor = tensor / np.linalg.norm(tensor, axis=1)[:, np.newaxis]
        angles = np.dot(tensor, tensor.T)
        logger.debug('angles matrix is {}'.format(angles.shape))
        np.fill_diagonal(angles, np.nan)
        return np.nanmean(angles)

    def orthosmplabs(self, tensor, name=None, iter=None, axis=-1):
        tensor = tensor.reshape(tensor.shape[0], -1)
        shape1, shape2 = tensor.shape
        # print tensor.shape
        tensor = tensor / np.linalg.norm(tensor, axis=1)[:, np.newaxis]
        angles = np.abs(np.dot(tensor, tensor.T))
        logger.debug('angles matrix is {}'.format(angles.shape))
        np.fill_diagonal(angles, np.nan)
        return np.nanmean(angles)

    def orthosmpl(self, tensor, name=None, iter=None, axis=-1):
        tensor = tensor.reshape(tensor.shape[0], -1)
        shape1, shape2 = tensor.shape
        # print tensor.shape
        tensor = tensor / np.linalg.norm(tensor, axis=1)[:, np.newaxis]
        angles = np.dot(tensor, tensor.T)
        logger.debug('angles matrix is {}'.format(angles.shape))
        np.fill_diagonal(angles, np.nan)
        return np.nanmean(angles)

    @utils.timeit('act ptrate consume')
    def ptrate(self, tensor, name, iter, win_size, thresh):
        mode = 'save'
        _iter, _val = self.ptrate_inst.pt_rate(tensor, name=name, iter=iter, win_size=win_size, thresh=thresh,
                                               mode=mode)

        if self.log_pnt[iter] >= 1 \
                and np.in1d(np.arange(iter - win_size + 1, iter + 1, step=1), self.log_pnt.index).all():
            mode = 'load'
            _iter, _val = self.ptrate_inst.pt_rate(tensor, name=name, iter=iter, win_size=win_size, thresh=thresh,
                                                   mode=mode)
            if _iter != iter - win_size // 2:
                logger.error('should log to right iter! {} {}'.format(_iter, iter - win_size // 2))
        return _iter, _val


class Windows(object):
    def __init__(self, max_win_size):
        self.l_tensor = {}
        self.l_iter = Queue.deque(maxlen=max_win_size)
        self.max_win_size = max_win_size

    def isfull(self, name, win_size):
        if name not in self.l_tensor:
            return False  # NOKEY
        elif len(self.l_tensor[name]) >= win_size:
            return True
        else:
            return False

    def include(self, tensor, iter, name, win_size):
        if not self.l_iter or self.l_iter[-1] != iter:
            self.l_iter.append(iter)
        if name not in self.l_tensor:
            self.l_tensor[name] = Queue.deque(maxlen=self.max_win_size)
            self.l_tensor[name].append(tensor)
        else:
            top = self.l_tensor[name][-1]
            if not (top is tensor):
                self.l_tensor[name].append(tensor)
            assert len(self.l_tensor[name]) == len(self.l_iter)

    def get_tensor(self, name, win_size):
        # import gc
        # gc.collect()
        # todo quite slow
        # cache = np.array(list(self.l_tensor[name])[-max_win_size:])
        # res_ = cache[-win_size:]
        res_ = np.array(list(self.l_tensor[name])[-win_size:])
        return res_

    def get_iter(self, win_size):
        _queue = self.l_iter
        return _queue[- (win_size // 2) - 1]


class TotVar(object):
    def __init__(self, windows):
        self.windows = windows

    def tot_var(self, tensor, iter, name, win_size, mode):
        self.windows.include(tensor, iter, name, win_size)
        if mode == 'load':
            if not self.windows.isfull(name, win_size=win_size):
                return NAN, NAN
            else:
                fenmu, sum = 0., 0.
                for ind in range(len(self.windows.l_tensor[name]) - 1):
                    last_tensor = self.windows.l_tensor[name][ind]
                    now_tensor = self.windows.l_tensor[name][ind + 1]
                    diff = np.abs(last_tensor - now_tensor)
                    sum += diff.mean()
                    fenmu += 1.
                sum /= fenmu
                logger.debug('layer {} iter {} totvar is {} '.format(name, iter, sum))
                return self.windows.get_iter(win_size), sum
        else:
            return NAN, NAN


class PTRate(object):
    def __init__(self, windows):
        self.windows = windows

    def pt_rate(self, tensor, iter, name, win_size, thresh, mode):

        self.windows.include(tensor, iter, name, win_size)
        if mode == 'load':
            if not self.windows.isfull(name, win_size=win_size):
                return NAN, NAN
            else:
                polarity = []
                for ind in range(len(self.windows.l_tensor[name]) - 1):
                    last_tensor = self.windows.l_tensor[name][ind]
                    now_tensor = self.windows.l_tensor[name][ind + 1]
                    polarity_now = ((last_tensor * now_tensor) < 0).astype(float)
                    polarity.append(polarity_now)
                polarity_time_space = np.array(polarity)
                polarity_time_space = polarity_time_space.reshape(polarity_time_space.shape[0], -1)

                polarity_space = polarity_time_space.mean(axis=0)

                if thresh == 'mean':
                    res = polarity_space.mean()
                    logger.debug('layer {} iter {} ptrate mean is {} '.format(name, iter, res))

                else:
                    _, res = thresh_proportion(arr=polarity_space, thresh=thresh)
                return self.windows.get_iter(win_size), res
        else:
            return NAN, NAN


class OnlineStd(object):
    class _OnlineStd(object):
        """
        Welford's algorithm computes the sample variance incrementally.
        """

        def __init__(self, iterable=None, ddof=1):
            self.ddof, self.n, = ddof, 0

            if iterable is not None:
                for datum in iterable:
                    self.include(datum)

        def include(self, datum):
            if self.n == 0:
                self.n += 1
                self.mean, self.M2 = np.zeros_like(datum, dtype=np.float), \
                                     np.zeros_like(datum, dtype=np.float)
            else:
                self.n += 1
                self.delta = datum - self.mean
                self.mean += self.delta / self.n
                self.M2 += self.delta * (datum - self.mean)

        @property
        def variance(self):
            if self.n < 2:
                return float('nan')
            else:
                return self.M2 / (self.n - self.ddof)

        @property
        def std(self):
            if self.n < 2:
                return float('nan')
            else:
                return np.sqrt(self.variance)

    def __init__(self, interval):
        self.last_std = {}
        self.interval = interval
        self.last_iter = {}
        self.record = {}
        self.record_min = {}
        self.df = pd.DataFrame()

    def online_std(self, tensor, iter, name, how='mean'):  # todo no how any more
        if name not in self.last_iter or iter - self.last_iter[name] >= self.interval:
            if name not in self.last_std:
                self.last_std[name] = self._OnlineStd()
            self.last_iter[name] = iter
            self.last_std[name].include(tensor)

            if not np.isnan(self.last_std[name].std).any():
                # # method 1
                # record_ = np.argmax(self.last_std[name].std).astype(int)
                # if len(self.record.get(name, set())) < 100:
                #   self.record[name] = self.record.get(name, set())
                #   self.record[name].add(record_)
                # else:
                #   logger.info('stdtime max capacity is 100')
                #
                # record_ = np.argmin(self.last_std[name].std).astype(int)
                # if len(self.record_min.get(name, set())) < 100:
                #   self.record_min[name] = self.record_min.get(name, set())
                #   self.record_min[name].add(record_)
                # else:
                #   logger.info('stdtime min capacity is 100')
                # # method 2
                # cap = 100
                # self.record[name] = self.last_std[name].std.ravel().argsort()[-cap:][::-1]
                # self.record_min[name] = self.last_std[name].std.ravel().argsort()[-cap:][::-1]
                # # method 3
                # if name not in self.record:
                #   self.record[name] = np.random.randint(0, tensor.ravel().shape[0], (100,))
                # if name not in self.record_min:
                #   self.record_min[name] = np.random.randint(0, tensor.ravel().shape[0], (100,))
                # for ind,record_ in enumerate(self.record[name]):
                #   self.df.loc[iter, name + '/example-max/' + str(ind)] = tensor.ravel()[record_]
                #
                # for ind,record_ in enumerate(self.record_min[name]):
                #   self.df.loc[iter, name + '/example-min/' + str(ind)] = tensor.ravel()[record_]
                pass

        if how == 'mean':
            stdtime_ = np.mean(self.last_std[name].std)
        else:
            stdtime_ = self.last_std[name].std

        return stdtime_


class Diff(object):
    def __init__(self, interval):
        self.interval = interval
        self.last_tensor = {}
        self.last_iter = {}

    def diff(self, tensor, iter, name):
        res = NAN
        if name not in self.last_iter or iter - self.last_iter[name] >= self.interval:
            if name in self.last_tensor:
                res = np.abs(tensor - self.last_tensor[name]).mean()
            self.last_iter[name] = iter
            self.last_tensor[name] = tensor
        return res


class OnlineTotVar(object):
    pass


class Histogram(object):
    pass


def fake_data(max_win_size, epochs, iter_per_epoch):
    series = pd.Series(data=3,
                       index=np.arange(epochs) * iter_per_epoch)
    series1 = pd.Series()
    for (ind0, _), (ind1, _) in zip(series.iloc[:-1].iteritems(), series.iloc[1:].iteritems()):
        if ind0 < 30 * iter_per_epoch:
            sample_rate = 4
        elif ind0 < 100 * iter_per_epoch:
            sample_rate = 2
        else:
            sample_rate = 1

        series1 = series1.append(
            pd.Series(data=2,
                      index=np.linspace(ind0, ind1, sample_rate, endpoint=False)[1:].astype(int))
        )

    series = series.append(series1)
    series1 = pd.Series()
    for ind, _ in series.iteritems():
        series1 = series1.append([
            pd.Series(data=1, index=np.arange(ind - max_win_size // 2, ind)),
            pd.Series(data=1, index=np.arange(ind + 1, ind + max_win_size // 2 + 1)),
        ])
    log_pnts = series.append(series1).sort_index()
    log_pnts.index = - log_pnts.index.min() + log_pnts.index
    hist_ = np.zeros(log_pnts.index.max() + 1)
    hist_[log_pnts.index] = log_pnts
    # from vis import *
    # plt.stem(hist)
    # plt.show()

    print log_pnts.shape
    return log_pnts


if __name__ == '__main__':
    epochs = 12
    iter_per_epoch = 150
    max_win_size = 11
    log_pnts = fake_data(max_win_size, epochs, iter_per_epoch)

    res = []
    act_stat = ActStat(max_win_size=max_win_size, log_pnt=log_pnts)
    kernel_stat = KernelStat(max_win_size=max_win_size, log_pnt=log_pnts)

    for _ind in range(epochs * iter_per_epoch):
        v = np.random.randn(128, 3, 3, 128) * (_ind + 1) / 100.
        _res = kernel_stat.calc_all(v, 'ok/kernel', _ind)
        res.append(_res)
        v = np.random.randn(6, 8, 8, 7) * (_ind + 1) / 100.
        _res = act_stat.calc_all(v, 'ok/act', _ind)
        res.append(_res)

    df = pd.concat(res)
    df.head()
    df = df.groupby(df.index).sum()
    df.head()

    # print df['ok/kernel/orthogonality']
    # print df['ok/act/orthogonality']
