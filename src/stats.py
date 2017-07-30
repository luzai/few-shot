import numpy as np


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

    def calc_all(self, tensor, name):
        res = {}
        for key, val in self.stat.iteritems():
            res[name + '/' + key] = val(self, tensor)
        return res

    def mag_mean(self, tensor):
        return np.abs(tensor).mean()


class Diff(object):
    def __init__(self):
        self.l_iter = {}
        self.l_tensor = {}

    def diff(self, tensor, iter, name):
        # mean
        # todo iter 12 continue
        res = {}
        if iter == self.l_iter[name]:
            res[name + '/' + 'diff'] = (tensor - self.l_tensor[name]).abs().mean()
        self.l_iter[name] = iter
        self.l_tensor[name] = tensor
        return res


class OnlineStd(object):
    def __init__(self):
        self.l_std = {}

    def online_std(self, tensor, iter, name):
        # todo sample per 100 make sure
        if name not in self.l_std:
            self.l_std[name] = _OnlineStd()
        assert iter % 100 == 0, 'todo'
        self.l_std[name].include(tensor)
        return {name + '/' + 'online_std': self.l_std[name].std}


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
            self.mean, self.M2 = np.zeros_like(datum, dtype=np.float)
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
