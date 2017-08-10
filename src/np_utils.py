from logs import logger
from configs import Config
from datasets import Dataset
import time, utils, glob, os, re, copy
import numpy as np, os.path as osp, pandas as pd, matplotlib.pylab as plt


class MultiIndexFacilitate(object):
    def __init__(self, columns):
        levels = list(columns.levels)
        names = list(columns.names)
        name2level = {name: level for name, level in zip(names, levels)}
        name2ind = {name: ind for ind, name in enumerate(names)}
        self.levels = levels
        self.names = names
        self.labels = columns.labels
        self.names2levels = name2level
        self.names2ind = name2ind
        self.index = columns

    def update(self):
        self.index = pd.MultiIndex.from_product(self.levels, names=self.names)


def get_shape(arr):
    return np.array(arr).shape


def dict2df(my_dict):
    tensor_d = {}
    for k, v in my_dict.iteritems():
        #     print k,v.shape
        if k[1] not in tensor_d:
            tensor_d[k[1]] = pd.Series(name=k[1], index=pd.Int64Index([]))
        tensor_d[k[1]][k[0]] = v
    return pd.DataFrame.from_dict(tensor_d)


def grid_iter(my_dict):
    names = my_dict.keys()
    levels = my_dict.values()
    import random
    columns = list(pd.MultiIndex.from_product(levels, names=names))
    random.shuffle(columns)
    for column in columns:
        yield dict(zip(names, column))


def df2arr(df):
    df = df.copy()
    df = df.unstack()
    shape = map(len, df.index.levels)
    arr = np.full(shape, np.nan)
    arr[df.index.labels] = df.values.flat
    return arr, MultiIndexFacilitate(df.index)


def arr2df(arr, indexf):
    df2 = pd.DataFrame(arr.flatten(), index=indexf.index)
    df2.reset_index().pivot_table(values=0, index=indexf.names[-1:], columns=indexf.names[:-1])
    return df2


if __name__ == '__main__':
    # from vis import Visualizer
    # visualizer = Visualizer(join='outer', stat_only=True, paranet_folder='stat401_10')
    # df = visualizer.perf_df
    # arr, indexf = df2arr(df)
    # print df.shape, arr.shape
    # df2 = arr2df(arr, indexf)

    grids = {'dataset': ['cifar10', ],
             'model_type': ['vgg10', ],
             'lr': np.logspace(-2, -3, 2),
             'queue': [None]
             }
    for iter in grid_iter(grids):
        print iter
