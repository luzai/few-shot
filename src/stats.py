import numpy as np
from logs import logger
import utils
import Queue


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


class Stat(object):
    def __init__(self, stat='all'):
        self.diff_inst = Diff()
        self.stdtime_inst = OnlineStd()
        if stat == 'all':
            self.stat = dict(Stat.__dict__)
            for key in self.stat.keys():
                if '_' in key:
                    del self.stat[key]

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
        name = kwargs.get('name')
        # if (tensor > 0.).all():
        #     logger.error(' bias all positive? name {}'.format(name))
        # elif (tensor < 0.).all():
        #     logger.error('bias all neg? name {}'.format(name))
        return tensor[tensor < 0].mean()

    def posproportion(self, tensor, **kwargs):
        pos_len = float(tensor[tensor > 0].shape[0])
        neg_len = float(tensor[tensor < 0.].shape[0])
        res = pos_len / (pos_len + neg_len)
        _, res2 = thres_proportion(tensor, 0.)
        assert np.isclose(res2, np.array([res])), 'different method calc pso_proportion should close'
        return res

    def calc_all(self, tensor, name):
        res = {}
        for key, val in self.stat.iteritems():
            res[name + '/' + key] = val(self, tensor, name=name)
        return res

    def magmean(self, tensor, **kwargs):
        return np.abs(tensor).mean()

    def sparsity(self, tensor, **kwargs):
        tensor = tensor.flatten()
        mean = tensor.mean()  # todo median or mean
        thresh = mean / 10.
        return float((tensor[tensor < thresh]).shape[0]) / float(tensor.shape[0])
        # for bias we usually set lr mult=0.1? --> bias is not sparse


class KernelStat(Stat):
    def __init__(self):
        super(KernelStat, self).__init__()
        _stat = dict(KernelStat.__dict__)
        for key in _stat.keys():
            if '_' in key:
                del _stat[key]
        self.stat = utils.dict_concat([self.stat, _stat])
        self.totvar_inst =TotVar()

    def orthogonality(self, tensor, name=None, iter=None):
        tensor = tensor.reshape(-1, tensor.shape[-1])
        angle = np.zeros((tensor.shape[-1], tensor.shape[-1]))
        it = np.nditer(angle, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            it[0] = angle_between(tensor[:, it.multi_index[0]], tensor[:, it.multi_index[1]])
            it.iternext()
        return angle.mean()
    def totvar(self,tensor,name=None,iter=None):
        return self.totvar_inst.tot_var(tensor,iter,name)



class BiasStat(Stat):
    def __init__(self):
        super(BiasStat, self).__init__()
        _stat = dict(BiasStat.__dict__)
        for key in _stat.keys():
            if '_' in key:
                del _stat[key]
        self.stat = utils.dict_concat([self.stat, _stat])
        self.totvar_inst = TotVar()
    def totvar(self,tensor,name=None,iter=None):
        return self.totvar_inst.tot_var(tensor, iter, name)


class ActStat(Stat):
    def __init__(self):
        super(ActStat, self).__init__()
        _stat = dict(ActStat.__dict__)
        for key in _stat.keys():
            if '_' in key:
                del _stat[key]
        self.stat = utils.dict_concat([self.stat, _stat])

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

    def ptrate(self):
        pass

# todo tot var & pt rate should share mem
class Windows(object):
    def __init__(self, win_size):
        self.l_tensor = {}
        self.l_iter = Queue.deque(maxlen=win_size)
        self.win_size = win_size

    def isfull(self, name):
        if name in self.l_tensor and self.l_tensor[name].shape[0] >= self.win_size:
            return True
        else:
            return False

    def include(self, tensor, iter, name):
        self.l_iter.append(iter)
        tensor = tensor[np.newaxis, ...]
        if not self.isfull(name):
            if name not in self.l_tensor:
                self.l_tensor[name] = tensor
            else:

                self.l_tensor[name] = np.concatenate((self.l_tensor[name], tensor),
                                                     axis=0)
        else:
            self.l_tensor[name] = np.concatenate((self.l_tensor[name], tensor), axis=0)
            self.l_tensor[name] = self.l_tensor[name][1:]

    def get_iter(self):
        _queue = list(self.l_iter)
        return _queue[len(_queue) // 2]


class Diff(Windows):
    def __init__(self):
        super(Diff, self).__init__(win_size=2)

    def diff(self, tensor, iter, name):
        self.include(tensor, iter, name)
        if self.isfull(name):
            res = np.abs(tensor - self.l_tensor[name]).mean()
        else:
            res = float('nan')
        return res


class TotVar(Windows):
    def __init__(self, win_size=11):
        super(TotVar, self).__init__(win_size=win_size)

    def tot_var(self, tensor, iter, name):
        self.include(tensor, iter, name)
        if not self.isfull(name):
            return None, None
        else:
            _tensor = self.l_tensor[name]
            _diff = np.abs(_tensor[1:] - _tensor[:-1])
            return self.get_iter(), _diff.mean()


class PTRate(Windows):
    def __init__(self, win_size, thresh=None):
        super(PTRate, self).__init__(win_size=win_size)
        self.thresh = thresh

    def pt_rate(self, tensor, iter, name):
        self.include(tensor, iter, name)
        if not self.isfull(name):
            return None, None
        else:
            _tensor = self.l_tensor[name]
            _tensor = _tensor.reshape(_tensor.shape[0], -1)
            polarity_time_space = np.sign(_tensor[1:] * _tensor[:-1])
            polarity_space = polarity_time_space.mean(0)
            if self.thresh is None:
                res = polarity_space.mean()
            else:
                _, res = thres_proportion(arr=polarity_space, thresh=self.thresh)
            return self.get_iter(), res


def thres_proportion(arr, thresh):
    hist, _ = np.histogram(arr, [arr.min(), thresh, arr.max() + np.finfo(float).eps])
    hist = hist / hist.astype(float).sum()
    assert np.allclose(hist.sum(), [1.]), 'histogram sum close to 1.'
    return hist


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

    def __init__(self):
        self.l_std = {}

    def online_std(self, tensor, iter, name):
        # todo sample per 100 make sure
        if name not in self.l_std:
            self.l_std[name] = self._OnlineStd()
        # assert iter % 100 == 0, 'todo'
        self.l_std[name].include(tensor)
        return np.mean(self.l_std[name].std)


class OnlineTotVar(object):
    pass


class Histogram(object):
    pass


if __name__ == '__main__':
    param_stat = KernelStat()
    v_l = []
    for _i in range(100):
        v = np.random.randn(10, 10, 10, 10)
        v_l.append(v)
        res = param_stat.calc_all(v, 'ok')
        # print res['ok/stdtime']
        _v = np.stack(v_l, axis=0)
        # print _v.std(axis=0).mean(), '\n'
        # print res['ok/orthogonality']
        print res
