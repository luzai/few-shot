import numpy as np, Queue, pandas as pd
from logs import logger
import utils
import math

NOKEY = 200

NAN = float('nan')


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_name2fn(class_name):
    return {key: val
            for key, val in class_name.__dict__.iteritems()
            if '_' not in key}


def thresh_proportion(arr, thresh):
    thresh = max(thresh, arr.min())
    thresh = min(thresh, arr.max())
    hist, _ = np.histogram(arr, [arr.min(), thresh, arr.max() + np.finfo(float).eps])
    hist = hist / hist.astype(float).sum()
    assert np.allclose(hist.sum(), [1.]), 'histogram sum close to 1.'
    return hist


# todo piandu fengdu

class Stat(object):
    def __init__(self, max_win_size, log_pnt):
        self.common_interval =  common_interval = np.diff(log_pnt)
        self.log_pnt = log_pnt
        self.stdtime_inst = OnlineStd(common_interval)
        self.diff_inst = Diff(common_interval)

        self.window = windows = Windows(win_size=max_win_size)
        self.totvar_inst = TotVar(windows)

        self.stat = get_name2fn(Stat)

    def diff(self, tensor, name=None, iter=None):
        return self.diff_inst.diff(tensor, iter, name)

    def stdtime(self, tensor, name=None, iter=None):
        return self.stdtime_inst.online_std(tensor, iter, name)

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

    def posmean(self, tensor, **kwargs):
        return tensor[tensor > 0].mean()

    def negmean(self, tensor, **kwargs):
        # todo do not observe softmax
        name = kwargs.get('name')
        # if (tensor > 0.).all():
        #     logger.error(' softmax all positive? name {}'.format(name))
        # elif (tensor < 0.).all():
        #     logger.error('softmax all neg? name {}'.format(name))
        return tensor[tensor < 0].mean()

    def posproportion(self, tensor, **kwargs):
        pos_len = float(tensor[tensor > 0].shape[0])
        neg_len = float(tensor[tensor < 0.].shape[0])
        res = pos_len / (pos_len + neg_len)
        _, res2 = thresh_proportion(tensor, 0.)
        if not abs(res2 - res) < np.finfo(float).eps:
            print 'different method calc pso_proportion should close {} {}'.format(res, res2)
        return res

    def magmean(self, tensor, **kwargs):
        return np.abs(tensor).mean()

    def sparsity(self, tensor, **kwargs):
        tensor = tensor.flatten()
        mean = tensor.mean()  # todo median or mean
        thresh = mean / 10.
        return float((tensor[tensor < thresh]).shape[0]) / float(tensor.shape[0])
        # for bias we usually set lr mult=0.1? --> bias is not sparse

    def totvar(self, tensor, name, iter, win_size):
        return self.totvar_inst.tot_var(tensor, iter, name, win_size)

    def calc_all(self, tensor, name, iter):
        # logger.info('calc tensor shape {} name {} iter {}'.format(tensor.shape,name,iter))
        res = pd.DataFrame()
        shield = ['totvar', 'ptrate']
        if iter in
        for fn_name, fn in self.stat.iteritems():
            if fn_name in shield: continue
            _name = name + '/' + fn_name
            _val = fn(self, tensor, name=name, iter=iter)
            res.loc[iter, _name] = _val

        fn_name = 'totvar'
        if fn_name in self.stat:
            for win_size in [11, 101]:
                _name = name + '/' + fn_name + '_win_size_' + str(win_size)
                _iter, _val = self.totvar(name=name, iter=iter, tensor=tensor, win_size=win_size)
                _iter = iter if math.isnan(_iter) else _iter
                res.loc[_iter, _name] = _val

        fn_name = 'ptrate'
        if fn_name in self.stat:
            for thresh in [.2, .6, 'mean']:
                for win_size in [11, 101]:
                    _name = name + '/' + fn_name + '_win_size_' + str(win_size) + '_thresh_' + str(thresh)
                    _iter, _val = self.ptrate(name=name, iter=iter, tensor=tensor, win_size=win_size, thresh=thresh)
                    _iter = iter if math.isnan(_iter) else _iter
                    res.loc[_iter, _name] = _val

        return res


class KernelStat(Stat):
    def __init__(self, max_win_size, log_pnt):
        super(KernelStat, self).__init__(max_win_size=max_win_size, log_pnt=log_pnt)
        _stat = dict(KernelStat.__dict__)
        for key in _stat.keys():
            if '_' in key:
                del _stat[key]
        self.stat = utils.dict_concat([self.stat, _stat])
        self.totvar_inst = TotVar(self.window)

    # todo for fc kernel different!
    def orthogonality(self, tensor, name=None, iter=None):
        tensor = tensor.reshape(-1, tensor.shape[-1])
        angle = np.zeros((tensor.shape[-1], tensor.shape[-1]))
        it = np.nditer(angle, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            it[0] = angle_between(tensor[:, it.multi_index[0]], tensor[:, it.multi_index[1]])
            it.iternext()
        return angle.mean()


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

    def orthogonality(self, tensor, name=None, iter=None):
        if len(tensor.shape) == 2:
            pass
        elif len(tensor.shape):
            pass
        tensor = tensor.reshape(tensor.shape[0], -1)
        angle = np.zeros((tensor.shape[0], tensor.shape[0]))
        it = np.nditer(angle, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            it[0] = angle_between(tensor[it.multi_index[0], :], tensor[it.multi_index[1], :])
            it.iternext()

        return angle.mean()

    def ptrate(self, tensor, name, iter, win_size, thresh):
        return self.ptrate_inst.pt_rate(tensor, name=name, iter=iter, win_size=win_size, thresh=thresh)


class Windows(object):
    def __init__(self, win_size):
        self.l_tensor = {}
        self.l_iter = Queue.deque(maxlen=win_size)
        self.win_size = win_size

    def isfull(self, name, win_size):
        if name not in self.l_tensor:
            return False  # NOKEY
        elif len(self.l_tensor[name]) >= win_size:
            return True
        else:
            return False

    def include(self, tensor, iter, name):
        if not self.l_iter or self.l_iter[-1] != iter:
            self.l_iter.append(iter)
        if name not in self.l_tensor:
            self.l_tensor[name] = Queue.deque(maxlen=self.win_size)
            self.l_tensor[name].append(tensor)
        else:
            top = self.l_tensor[name][-1]
            if not (top is tensor or np.array_equal(top, tensor)):
                self.l_tensor[name].append(tensor)

                # todo mem recycle

    def get_tensor(self, name, win_size):
        return np.array(list(self.l_tensor[name])[-win_size:])

    def get_iter(self):
        _queue = self.l_iter
        return _queue[len(_queue) // 2]


class TotVar(object):
    def __init__(self, windows):
        self.windows = windows

    def tot_var(self, tensor, iter, name, win_size):
        self.windows.include(tensor, iter, name)
        if not self.windows.isfull(name, win_size=win_size):
            return NAN, NAN
        else:
            _tensor = self.windows.get_tensor(name, win_size)
            _diff = np.abs(_tensor[1:] - _tensor[:-1])
            return self.windows.get_iter(), _diff.mean()


class PTRate(object):
    def __init__(self, windows):
        self.windows = windows

    def pt_rate(self, tensor, iter, name, win_size, thresh):
        self.windows.include(tensor, iter, name)
        if not self.windows.isfull(name, win_size=win_size):
            return NAN, NAN
        else:
            _tensor = self.windows.get_tensor(name, win_size)
            _tensor = _tensor.reshape(_tensor.shape[0], -1)
            polarity_time_space = (-np.sign(_tensor[1:] * _tensor[:-1]) + 1.) / 2.
            polarity_space = polarity_time_space.mean(axis=0)
            if thresh == 'mean':
                res = polarity_space.mean()
            else:
                _, res = thresh_proportion(arr=polarity_space, thresh=thresh)
            return self.windows.get_iter(), res


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

    def online_std(self, tensor, iter, name):
        if name not in self.last_iter or self.last_iter[name] + self.interval == iter:
            if name not in self.last_std:
                self.last_std[name] = self._OnlineStd()
            self.last_iter[name] = iter
            self.last_std[name].include(tensor)
        return np.mean(self.last_std[name].std)


class Diff(object):
    def __init__(self, interval):
        self.interval = interval
        self.last_tensor = {}
        self.last_iter = {}

    def diff(self, tensor, iter, name):
        res = NAN
        if name not in self.last_iter or self.last_iter[name] + self.interval == iter:
            if name in self.last_tensor:
                res = np.abs(tensor - self.last_tensor[name]).mean()
            self.last_iter[name] = iter
            self.last_tensor[name] = tensor
        return res


class OnlineTotVar(object):
    pass


class Histogram(object):
    pass


if __name__ == '__main__':
    epochs = 301
    iter_per_epoch = 196
    max_win_size = 11
    log_freq = np.array([4, 2, 1])
    log_freq_bin = np.array([0, epochs // 10, epochs // 3, epochs])

    log_freq_bin_iter = log_freq_bin * iter_per_epoch
    # log_freq_bin_iter[0] = self.max_win_zise//2
    log_num = np.diff(log_freq_bin) * log_freq
    log_pnt = np.concatenate([
        np.linspace(start, stop, num, endpoint=False).astype(int)
        for num, start, stop in
        zip(log_num, log_freq_bin_iter[:-1], log_freq_bin_iter[1:])
    ])
    log_pnts_l = []
    for pnt in log_pnt:
        _arr = np.arange(pnt - int(max_win_size // 2),
                         pnt + int(max_win_size // 2) + 1).astype(int)
        if (_arr < 0).any():
            _arr += - _arr.min()
        log_pnts_l.append(_arr)
    log_pnts = np.concatenate(log_pnts_l)
    print log_pnts.shape

    kernel_stat = KernelStat(max_win_size=max_win_size, log_pnt=196)
    print kernel_stat.stat
    v_l = []
    res = []
    from utils import timer

    timer.tic()
    for _ind in range(196 * 2 + 1):
        v = np.random.randn(3, 3, 3, 32) * (_ind + 1) / 100.
        v_l.append(v)
        if _ind in log_pnts:
            res.append(kernel_stat.calc_all(v, 'ok/kernel', _ind))
            # print timer.toc()
        _v = np.stack(v_l, axis=0)
        # assert np.allclose(res['ok/kernel/stdtime'], _v.std(axis=0).mean()), ' should close '
    res_df = pd.DataFrame(index=np.arange(196 * 2 + 1), columns=res[0].columns)
    for _df in res[1:]:
        res_df.update(_df)
    res_df.dropna(how='all', inplace=True)

    # res = []
    # act_stat = ActStat()
    # print act_stat.stat
    # for _ind in range(102):
    #     v = np.random.randn(10, 10, 10, 10)
    #     _res = act_stat.calc_all(v, 'ok/act', _ind)
    #     res.append(_res)
    #     print _res.shape
    # res = pd.concat(res, axis=0)
    #
    # res = []
    # bias_stat = BiasStat()
    # print bias_stat.stat
    #
    # for _ind in range(102):
    #     v = np.random.randn(10, 10, 10, 10)
    #     res.append(bias_stat.calc_all(v, 'ok/bias', _ind))
    #
    # res = pd.concat(res, axis=0)
    print 'ok'
