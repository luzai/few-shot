from loader import Loader
from log import logger
from opts import Config
import time, numpy as np
import pandas as pd, re, copy
import time, glob, os, os.path as osp
from datasets import Dataset
from models import VGG
from opts import Config
from log import logger
import matplotlib

matplotlib.use('TkAgg')
matplotlib.style.use('ggplot')
import matplotlib.pylab as plt


def drop_level(perf_df, other_name=None):
    columns = perf_df.columns
    names = columns.names
    names = np.array(names)
    levels = columns.levels
    name2level = {name: level for name, level in zip(names, levels)}
    res_str = 'Ttl_'
    if other_name is None:
        while True:
            levels_len = [len(level) for level in perf_df.columns.levels]
            levels_len = np.array(levels_len)
            if (levels_len != np.ones_like(levels_len)).all() or len(perf_df.columns.levels) == 3: break

            for ind, level in enumerate(perf_df.columns.levels):
                if len(level) == 1:
                    res_str += level.name + '_' + level[0] + '_'
                    perf_df.columns = perf_df.columns.droplevel(ind)
                    break
    else:
        perf_df.columns = perf_df.columns.droplevel(other_name)
        for name in other_name:
            level = name2level[name]
            res_str += str(name) + '_' + str(level[0]) + '_'
    return perf_df, res_str[:-1]


class Visualizer(object):
    def __init__(self, config_dict, join='inner', stat_only=True, paranet_folder='stat'):
        self.aggregate(config_dict, join, stat_only=stat_only, parant_folder=paranet_folder)
        self.split()
        # self.names2levels = {level: np.unique(self.columns.get_level_values(level)) for level in self.names}
        self.names2levels = {name: level for level, name in zip(self.df.columns.levels, self.df.columns.names)}

    def split(self):
        self.perf_df = self.select(self.df, 'name', "(?:val_loss|loss|val_acc|acc)")
        self.stat_df = self.select(self.df, 'name', "^obs.*")

    def _plot(self, perf_df, axes_names, legend):
        row_name, col_name, inside = axes_names
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

        fig, targets = plt.subplots(rows, cols, figsize=(18, 9))  # , sharex=True)
        if len(targets.shape) == 1: targets = targets.reshape(rows, cols)
        for _i in range(rows):
            try:
                targets[_i][0].set_ylabel(row_level[_i])
            except:
                from IPython import embed;
                embed()
        for _j in range(cols):
            targets[0][_j].set_title(col_level[_j])

        target = []
        legends = np.zeros((rows, cols)).astype(object)
        for inds in perf_df.columns:
            for ind in inds:
                if ind in row_level: _row = level2row[ind]
                if ind in col_level: _col = level2col[ind]
            for ind in inds:
                if ind in inside_level:
                    if legends[_row, _col] == 0:
                        legends[_row, _col] = [ind]
                    else:
                        legends[_row, _col] += [ind]
            target.append(targets[_row, _col])

        perf_df.plot(subplots=True, legend=False, ax=target, marker=None)

        for _row in range(legends.shape[0]):
            for _col in range(legends.shape[1]):
                if legend:
                    targets[_row, _col].legend(legends[_row, _col])
                else:
                    targets[_row, _col].legend([])

        return fig

    def plot(self, perf_df, axes_names, other_names=None, legend=True):
        # order of axes is (row,col,inside fig)
        perf_df, super_title = drop_level(perf_df, other_names)
        if len(perf_df.columns.names) != 3:
            from IPython import embed;
            embed()

        fig = self._plot(perf_df, axes_names, legend)
        fig.suptitle(super_title)
        fig.savefig(Config.output_path + '/' + super_title + '.png')
        return fig

    def select(self, df, level_name, par_name, sort_names=None, regexp=True):
        sel_name = df.columns.get_level_values(level_name)
        f_sel_name = set()
        pat = re.compile(par_name)
        for sel_name_ in sel_name:
            judge = bool(pat.match(sel_name_)) if regexp else par_name == sel_name_
            if judge: f_sel_name.add(sel_name_)
        df_l = []
        for sel_name_ in f_sel_name:
            df_l.append(df.xs(sel_name_, level=level_name, axis=1, drop_level=False))
        if df_l != []:
            df = pd.concat(df_l, axis=1)
        else:
            return None
        if sort_names is None: sort_names = df.columns.names
        df.sort_index(level=sort_names, axis=1, inplace=True)
        return df

    def aggregate(self, conf_dict_in, join, parant_folder, stat_only):
        conf_name_dict = {}
        loaders = {}
        for model_type in conf_dict_in.get('model_type', [None]):
            for lr in conf_dict_in.get('lr', [None]):
                for dataset_type in conf_dict_in.get('dataset_type', [None]):
                    conf = Config(epochs=231, batch_size=256, verbose=2,
                                  model_type=model_type,
                                  dataset_type=dataset_type,
                                  debug=False, others={'lr': lr},  # , 'limit_val': True
                                  clean=False)

                    path = Config.root_path + '/' + parant_folder + '/' + conf.name
                    if osp.exists(path):
                        _res = {}
                        for ind, val in conf.to_dict().items():
                            if isinstance(val, float):
                                _res[ind] = '{:.2e}'.format(val)
                            else:
                                _res[ind] = str(val)
                        conf_name_dict[conf.name] = _res
                        loader = Loader(path=path, stat_only=stat_only)
                        # loader.start()
                        loader.load(stat_only=stat_only)
                        loaders[conf.name] = loader
        df_l = []
        index_l = []
        assert len(conf_name_dict) != 0, 'should not be empty'
        for ind in range(len(conf_name_dict)):
            conf_name = conf_name_dict.keys()[ind]
            conf_dict = conf_name_dict[conf_name]

            loader = loaders[conf_name]
            # loader.join()
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
        df.index.name = 'epoch' if not stat_only else 'iter'
        self.df = df

    def auto_plot(self, df, ):
        columns = df.columns
        names = columns.names
        names = np.array(names)
        levels = columns.levels
        name2level = {name: level for name, level in zip(names, levels)}
        for axes_names in names[comb_index(len(names), 3)]:
            other_names = list(set(names) - set(axes_names))
            for poss in cartesian([name2level[name] for name in other_names]):
                _df = df.copy()
                for _name, _poss in zip(other_names, poss):
                    _df = self.select(_df, _name, _poss, regexp=False)
                    if _df is None:    break
                if _df is None: continue
                self.plot(_df, axes_names, other_names)
                plt.close()


from itertools import combinations, chain
from scipy.misc import comb


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def comb_index(n, k):
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                        int, count=count * k)
    return index.reshape(-1, k)


def expand_level(columns):
    levels = columns.levels
    names = columns.names
    fname = copy.deepcopy(names[:-1])
    for _ind, _name in enumerate(split_path(columns[0][-1])):
        fname += ['name' + str(_ind)]
    # finds = np.zeros((len(columns),len(fname)),basestring)
    finds = []
    for inds in columns:
        finds.append(list(inds[:-1]) + list(split_path(inds[-1])))
    finds = np.array(finds).astype(basestring).transpose()
    fcolumns = pd.MultiIndex.from_arrays(finds, names=list(fname))
    return fcolumns


def split_path(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break

    folders.reverse()
    return folders


if __name__ == '__main__':
    tic = time.time()
    config_dict = {'model_type': ['vgg6', 'vgg10', 'resnet6', 'resnet10'],
                   'lr': np.concatenate((np.logspace(0, -5, 6), np.logspace(-1.5, -2.5, 0))),
                   'dataset_type': ['cifar10', 'cifar100']
                   }
    visualizer = Visualizer(config_dict, join='inner', stat_only=True, paranet_folder='stat')

    # perf_df = visualizer.perf_df
    # visualizer.auto_plot(perf_df)
    # _df = perf_df
    # # _df = visualizer.select(_df, 'dataset_type', 'cifar10')
    # _df = visualizer.select(_df, 'model_type', 'vgg6')
    #
    # visualizer.plot(_df, ('dataset_type', 'name', 'lr'))
    # visualizer.plot(_df, ('lr', 'name', 'dataset_type'))
    print time.time() - tic

    df = stat_df = visualizer.stat_df
    df.columns = expand_level(df.columns)
    visualizer.auto_plot(df)
    # df = visualizer.select(df, 'name2', 'act')
    # df = visualizer.select(df, 'name1', 'conv2d')
    # df = visualizer.select(df, 'model_type', 'vgg6')
    # df = visualizer.select(df, 'dataset_type', 'cifar10$')
    # # df = visualizer.select(df, 'lr', '1.*?e-02')
    # # todo plot title
    # visualizer.plot_stat(df, ('name0', 'name3', 'lr'))

    plt.show()
