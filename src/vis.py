from loader import Loader
from log import logger
from opts import Config
import time, numpy as np
import pandas as pd, re
import time, glob, os, os.path as osp
from datasets import Dataset
from models import VGG
from opts import Config
from log import logger


class Visualizer(object):
    def __init__(self, config_dict, join='inner'):
        self.aggregate(config_dict, join)
        self.split()
        # self.names2levels = {level: np.unique(self.columns.get_level_values(level)) for level in self.names}
        self.names2levels = {name: level for level, name in zip(self.df.columns.levels, self.df.columns.names)}

    def split(self):
        self.perf_df = self.select(self.df, 'name', "(?:val_loss|loss|val_acc|acc)")
        self.stat_df = self.select(self.df, 'name', "^obs.*")

    def plot_perf(self, perf_df, axes_names):
        # order of axes is (row,col,inside fig)
        row_name, col_name, inside = axes_names

        while True:
            for ind, level in enumerate(perf_df.columns.levels):
                if len(level) == 1:
                    perf_df.columns = perf_df.columns.droplevel(ind)
                    break
            levels_len = [len(level) for level in perf_df.columns.levels]
            levels_len = np.array(levels_len)
            if (levels_len != np.ones_like(levels_len)).all(): break
        assert len(perf_df.columns.names) == 3

        names2levels = {name: level for level, name in zip(perf_df.columns.levels, perf_df.columns.names)}
        row_level = names2levels[row_name]
        level2row = {val: ind for ind, val in enumerate(row_level)}
        rows = len(row_level)

        col_level = names2levels[col_name]
        level2col = {val: ind for ind, val in enumerate(col_level)}
        cols = len(col_level)

        inside_level = names2levels[inside]
        level2inside = {val: ind for ind, val in enumerate(inside_level)}
        insides = len(inside_level)

        fig, targets = plt.subplots(rows, cols, sharex=True)
        for _i in range(rows):
            targets[_i][0].set_ylabel(row_level[_i])
        for _j in range(cols):
            targets[0][_j].set_title(col_level[_j])

        target = []
        legends = np.zeros((rows,cols)).astype(object)
        for inds in perf_df.columns:
            for ind in inds:
                if ind in row_level: _row = level2row[ind]
                if ind in col_level: _col = level2col[ind]
            for ind in inds:
                if ind in inside_level:
                    if legends[_row,_col] ==0: legends[_row,_col] = [ind]
                    else: legends[_row,_col] +=[ind]
            target.append(targets[_row,_col])

        perf_df.plot(subplots=True, legend=True, ax=target)

        for _row in range(legends.shape[0]):
            for _col in range(legends.shape[1]):
                targets[_row,_col].legend(legends[_row,_col])

        return fig

    def select(self, df, level_name, par_name):
        sel_name = df.columns.get_level_values(level_name)
        f_sel_name = set()
        pat = re.compile(par_name)
        for sel_name_ in sel_name:
            judge = bool(pat.match(sel_name_))
            if judge: f_sel_name.add(sel_name_)
        df_l = []
        for sel_name_ in f_sel_name:
            df_l.append(df.xs(sel_name_, level=level_name, axis=1, drop_level=False))
        df = pd.concat(df_l, axis=1)
        df.sort_index(level=self.names, axis=1, inplace=True)
        return df

    def aggregate(self, conf_dict_in, join):
        conf_name_dict = {}
        loaders = {}
        for model_type in conf_dict_in.get('model_type', [None]):
            for lr in conf_dict_in.get('lr', [None]):
                conf = Config(epochs=231, batch_size=256, verbose=2,
                              model_type=model_type,
                              dataset_type='cifar10',
                              debug=False, others={'lr': lr},  # , 'limit_val': True
                              clean=False)
                path = Config.root_path + '/epoch/' + conf.name
                if osp.exists(path):
                    _res = {}
                    for ind, val in conf.to_dict().items():
                        if isinstance(val, float):
                            _res[ind] = '{:.2e}'.format(val)
                        else:
                            _res[ind] = str(val)
                    conf_name_dict[conf.name] = _res
                    loader = Loader(path=path)
                    loader.load()
                    loaders[conf.name] = loader
        df_l = []
        index_l = []
        assert len(conf_name_dict) != 0, 'should not be empty'
        for ind in range(len(conf_name_dict)):
            conf_name = conf_name_dict.keys()[ind]
            conf_dict = conf_name_dict[conf_name]

            loader = loaders[conf_name]
            loader.load()

            scalar = loader.scalars
            act = loader.act
            param = loader.params

            df_l.append(scalar)
            for name in scalar.columns:
                index_l.append(conf_dict.values() + [name])

            df_l.append(act)
            for name in act.columns:
                index_l.append(conf_dict.values() + [name])

            df_l.append(param)
            for name in param.columns:
                index_l.append(conf_dict.values() + [name])

        index_l = np.array(index_l).astype(basestring).transpose()
        index_name = conf_dict.keys() + ['name']
        index = pd.MultiIndex.from_arrays(index_l, names=index_name)
        df = pd.concat(df_l, axis=1, join=join)
        df.columns = index
        df = df.sort_index(axis=1, level=index_name)
        self.names = index_name
        self.columns = index
        df.index.name = 'epoch'
        self.df = df

import  matplotlib
matplotlib.use('TkAgg')
matplotlib.style.use('ggplot')
import matplotlib.pylab as plt

if __name__ == '__main__':
    config_dict = {'model_type': ['vgg5', 'vgg11', 'vgg19'],
                   'lr': [1, 1e-2, 1e-5]}
    visualizer = Visualizer(config_dict)

    perf_df = visualizer.perf_df
    _df = perf_df
    _df = visualizer.select(_df, 'dataset_type', 'cifar10')

    visualizer.plot_perf(_df, ('model_type', 'name', 'lr'))

    plt.show()