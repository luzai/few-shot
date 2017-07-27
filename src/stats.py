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

    def calc_all(self,tensor,name):
        res={}
        for key,val in self.stat.iteritems():
            res[name+'/'+key]=val(self,tensor)
        return  res