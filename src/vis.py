import matplotlib

matplotlib.use('Agg')

from loader import Loader
from logs import logger
from configs import Config
from datasets import Dataset
from models import VGG

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from matplotlib import colors as mcolors

from sklearn import preprocessing, manifold, datasets

import time, utils, glob, os, re, copy
import numpy as np, os.path as osp, pandas as pd, matplotlib.pylab as plt

from itertools import combinations, chain
from scipy.misc import comb
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

matplotlib.style.use('ggplot')

Axes3D

dbg = False


def drop_level(perf_df, other_name=None, keep_num_levels=3):
    perf_df = perf_df.copy()
    columns = perf_df.columns
    names = columns.names
    names = np.array(names)
    levels = columns.levels
    name2level = {name: level for name, level in zip(names, levels)}

    res_str = ''
    if other_name is None:
        while True:
            levels_len = [len(level) for level in perf_df.columns.levels]
            levels_len = np.array(levels_len)
            if (levels_len != np.ones_like(levels_len)).all() or len(perf_df.columns.levels) == keep_num_levels: break

            for ind, level in enumerate(perf_df.columns.levels):
                if len(level) == 1:
                    res_str += level[0] + '_'
                    perf_df.columns = perf_df.columns.droplevel(ind)
                    break
    else:
        perf_df.columns = perf_df.columns.droplevel(other_name)
        for name in other_name:
            level = name2level[name]
            res_str += level[0] + '_'
    return perf_df, res_str


@utils.static_vars(ind=0, label2color={u'1.00e+00': u'#E24A33',
                                       u'1.00e-01': u'#348ABD',
                                       u'1.00e-02': u'#988ED5',
                                       u'1.00e-03': u'#777777',
                                       u'1.00e-04': u'#FBC15E',
                                       u'1.00e-05': u'#8EBA42'})
def get_colors(label):
    colors = [u'#FFFF00', u'#FF8C00', u'#FFEFD5', u'#FFA500', u'#6B8E23', u'#87CEEB', u'#006400', u'#008080',
              u'#9ACD32', u'#696969', u'#F0FFF0', u'#BDB76B', u'#FFE4C4', u'#F5DEB3', u'#4682B4', u'#800080',
              u'#F5F5DC', u'#FDF5E6', u'#A0522D', u'#00FF00', u'#00FA9A', u'#CD853F', u'#D3D3D3', u'#D2691E',
              u'#808000', u'#FFFAF0', u'#808080', u'#FF1493', u'#800000', u'#D2B48C', u'#DB7093', u'#B0E0E6',
              u'#191970', u'#9932CC', u'#90EE90', u'#FFF0F5', u'#008000']
    if label not in get_colors.label2color:
        color = colors[get_colors.ind]
        get_colors.ind += 1
        if get_colors.ind + 1 >= len(color):
            get_colors.ind = 0
        get_colors.label2color[label] = color
        logger.info('update label2color {}'.format(get_colors.label2color))
    else:
        color = get_colors.label2color[label]
    return color


class Visualizer(object):
    def __init__(self, config_dict, join='inner', stat_only=True, paranet_folder='stat'):
        self.aggregate(config_dict, join, stat_only=stat_only, parant_folder=paranet_folder)
        self.split()
        # self.names2levels = {level: np.unique(self.columns.get_level_values(level)) for level in self.names}
        self.names2levels = {name: level for level, name in zip(self.df.columns.levels, self.df.columns.names)}

    def split(self):
        self.perf_df = self.select(self.df, 'name', "(?:val_loss|loss|val_acc|acc)")
        self.stat_df = self.select(self.df, 'name', "^obs.*")

    def plot(self, perf_df, axes_names, other_names=None, legend=True):

        # #  order of axes is (row,col,inside fig)
        perf_df, sup_title = drop_level(perf_df, other_names)
        assert len(perf_df.columns.names) == 3, 'plot only accept input 3'

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
        from cycler import cycler

        # # arrange xlabel ylabel
        fig, axes = plt.subplots(rows, cols, figsize=(max(18 / 5. * cols, 18), 9 / 4. * rows))  # , sharex=True)
        axes = np.array(axes).reshape(rows, cols)

        for _i in range(rows):
            try:
                axes[_i][0].set_ylabel(row_level[_i])
            except Exception as inst:
                raise ValueError(str(inst))

        for _j in range(cols):
            axes[0][_j].set_title(col_level[_j])
        # # plot to right place
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
            target.append(axes[_row, _col])
        pd.DataFrame.plot()
        perf_df.plot(subplots=True, legend=False, ax=target, marker=None, sharex=False)
        #  # change color

        for axis in axes.flatten():
            for line in axis.get_lines():
                _label = line.get_label()
                for level in inside_level:
                    if level in _label: label = level
                color = get_colors(label)
                line.set_color(color)

        # # plot legend
        axes[0, 0].legend(list(legends[0, 0]))
        for _row in range(legends.shape[0]):
            for _col in range(legends.shape[1]):
                axes[_row, _col].yaxis.get_major_formatter().set_powerlimits((-2, 2))
                # if legend:
                #     axes[_row, _col].legend(list(legends[_row, _col]))
                # else:
                #     axes[_row, _col].legend([])

        sup_title += 'legend_' + inside
        fig.suptitle(sup_title)
        # plt.show()
        return fig, sup_title

    def select(self, df, level_name, par_name, sort_names=None, regexp=True):
        df = df.copy()
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
            act = act.round(7)
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

    def auto_plot(self, df, path_suffix, axes_names=None):

        columns = df.columns
        levels, names, name2level, name2ind = get_columns_alias(columns)
        show = False
        if axes_names is not None:
            axes_names_l = [axes_names]
        else:
            axes_names_l = choose_three(names)
        for axes_names in axes_names_l:
            other_names = list(set(names) - set(axes_names))
            for poss in cartesian([name2level[name] for name in other_names]):
                # try:
                _df = df.copy()
                for _name, _poss in zip(other_names, poss):
                    _df = self.select(_df, _name, _poss, regexp=False)
                    if _df is None: break
                if _df is None: continue
                fig, sup_title = self.plot(_df, axes_names, other_names)
                utils.mkdir_p(Config.output_path + path_suffix + '/')
                fig.savefig(Config.output_path + path_suffix + '/' + re.sub('/', '', sup_title) + '.png')
                if show: plt.show()
                plt.close()
                if globals()['dbg']:
                    logger.info('dbg mode break')
                    break
                    # except Exception as inst:
                    #     from IPython import embed
                    #     embed()
                    # print inst


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


def choose_three(names):
    def comb_index(n, k):
        assert k == 3, 'choose 3 dim'
        count = comb(n, k, exact=True)
        index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                            int, count=count * k)
        index = index.reshape(-1, k)
        index = np.concatenate((index, index[:, [1, 2, 0]], index[:, [0, 2, 1]]), axis=0)
        # np.random.shuffle(index)
        return index

    names = np.array(names)
    for axes_names in names[comb_index(len(names), 3)]:
        if axes_names[-1] != 'lr':
            continue
        else:
            yield axes_names


def expand_level(columns):
    levels, names, name2level, name2ind = get_columns_alias(columns)
    fname = copy.deepcopy(names[:-1])
    for _ind, _name in enumerate(split_path(columns[0][-1])):
        fname.append('name' + str(_ind))
    finds = []
    for inds in columns:
        finds.append(list(inds[:-1]) + list(split_path(inds[-1])))
    finds = np.array(finds).astype(basestring).transpose()
    fcolumns = pd.MultiIndex.from_arrays(finds, names=list(fname))
    return fcolumns


def get_columns_alias(columns):
    levels = columns.levels
    names = columns.names
    name2level = {name: level for name, level in zip(names, levels)}
    name2ind = {name: ind for ind, name in enumerate(names)}
    return list(levels), list(names), name2level, name2ind


def merge_level(columns, name1, name2):
    levels, names, name2level, name2ind = get_columns_alias(columns)
    ind1 = name2ind[name1]
    ind2 = name2ind[name2]
    assert ind1 + 1 == ind2
    fnames = names[:ind1] + [name1 + '/' + name2] + names[ind2 + 1:]

    finds = []
    for inds in columns:
        finds.append(list(inds[:ind1] + (inds[ind1] + '/' + inds[ind2],) + inds[ind2 + 1:]))
    finds = np.array(finds).astype(basestring).transpose()
    fcolumns = pd.MultiIndex.from_arrays(finds, names=list(fnames))
    return fcolumns


def split_path(path):
    folders = []
    while True:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break

    folders.reverse()
    return folders


def subplots(visualizer, path_suffix):
    # plot all performance
    perf_df = visualizer.perf_df.copy()
    if 'stable' in path_suffix:
        # columns = perf_df.columns
        # levels, names, name2level, name2ind = get_columns_alias(columns)
        perf_df = visualizer.select(perf_df, 'lr', '1.00e-0[2-9].*')

    # visualizer.auto_plot(perf_df, path_suffix + '_all_perf')
    # plot only val acc
    perf_df = visualizer.select(perf_df, 'name', 'val_acc$')
    visualizer.auto_plot(perf_df, path_suffix + '_val_acc')

    # plot statistics
    df = visualizer.stat_df.copy()

    if 'stable' in path_suffix:
        # columns = perf_df.columns
        # levels, names, name2level, name2ind = get_columns_alias(columns)
        df = visualizer.select(df, 'lr', '1.00e-0[2-9].*')

    df.columns = expand_level(df.columns)
    # # name0 1 2 3 -- obs0 conv2d act iqr
    df.columns = merge_level(df.columns, 'name0', 'name1')

    # visualizer.auto_plot(visualizer.select(df, 'name3', '(?:mean|median|iqr|std)'), path_suffix + '_std_iqr',
    #                      axes_names=('name0/name1', 'name3', 'lr'))

    visualizer.auto_plot(df, path_suffix + '_all_stat',
                         axes_names=('name0/name1', 'name3', 'lr'))


def t_sne(visualizer, model_type, dataset_type, start_lr):
    logger.info('start ' + '_'.join((model_type, dataset_type, 'start_lr', str(start_lr))))
    stat_df = visualizer.stat_df.copy()
    stat_df = visualizer.select(stat_df, 'lr', '1.00e[+-]0[{}-9].*'.format(start_lr))
    stat_df = visualizer.select(stat_df, 'name', '(?:.*/dense/.*|.*/conv2d/.*)')
    stat_df = visualizer.select(stat_df, 'dataset_type', dataset_type, regexp=False)
    stat_df = visualizer.select(stat_df, 'model_type', model_type, regexp=False)

    stat_df, suptitle = drop_level(stat_df, keep_num_levels=2)

    perf_df = visualizer.perf_df.copy()
    perf_df = visualizer.select(perf_df, 'lr', '1.00e[+-]0[{}-9].*'.format(start_lr))
    perf_df = visualizer.select(perf_df, 'dataset_type', dataset_type, regexp=False)
    perf_df = visualizer.select(perf_df, 'model_type', model_type, regexp=False)
    perf_df = visualizer.select(perf_df, 'name', 'val_acc', regexp=False)

    perf_df, suptitle = drop_level(perf_df, keep_num_levels=2)

    stats = []
    colors = []
    val_accs = []

    levels, names, name2level, name2ind = get_columns_alias(stat_df.columns)
    for ind, lr in enumerate(name2level['lr']):
        _perf_df = visualizer.select(perf_df, 'lr', lr, regexp=False)
        _perf_df, _suptitle = drop_level(_perf_df, other_name=['lr'],
                                         keep_num_levels=1)
        logger.debug ('{}'.format(_perf_df.columns))
        val_accs.append(_perf_df.as_matrix())

        _stat_df = visualizer.select(stat_df, 'lr', lr, regexp=False)
        _stat_df, _suptitle = drop_level(_stat_df, other_name=['lr'],
                                         keep_num_levels=1)
        colors.append(ind)
        stats.append(_stat_df.as_matrix())

    stats_all = np.concatenate(stats, axis=0)
    # colors = np.concatenate(colors, axis=0)
    # val_accs = np.concatenate(val_accs, axis=0)

    stats_all = preprocessing.scale(stats_all, axis=0)
    # for multitime in range(1):
    stats_dim2_all = manifold.TSNE(n_components=2).fit_transform(stats_all)

    stats_dim2 = []
    ind = 0
    for stat in stats:
        stats_dim2.append(stats_dim2_all[ind:ind + stat.shape[0]])
        ind += stat.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if globals()['dbg']:
        dur = 0.1
    else:
        dur = 9.
    freq = stats_dim2[0].shape[0] / dur
    has_legend = False

    def make_frame_mpl(t):
        ax.clear()
        time_i = int(t * freq)
        for ind, (stat_dim2, color, val_acc) in enumerate(zip(stats_dim2, colors, val_accs)):

            val_acc[:time_i].reshape((time_i,))
            lr = name2level['lr'][ind]
            ax.scatter(stat_dim2[:time_i, 0],
                       stat_dim2[:time_i, 1],
                       val_acc[:time_i].reshape((time_i,)),
                       c=get_colors(lr),
                       label=lr
                       # cmap=plt.cm.Spectral,
                       )

        ax.legend(loc='upper right')
        # ax.get_xaxis().set_ticklabels([])
        # ax.get_yaxis().set_ticklabels([])
        ax.set_xlim3d([stats_dim2_all[:, 0].min(), stats_dim2_all[:, 0].max()])
        ax.set_ylim3d([stats_dim2_all[:, 1].min(), stats_dim2_all[:, 1].max()])
        ax.set_zlim3d([0, 1])
        ax.view_init(elev=20., azim=t / dur * 130.)
        fig.suptitle('_'.join((model_type, dataset_type, 'start_lr', str(start_lr))))
        return mplfig_to_npimage(fig)
    ## animation
    # animation = mpy.VideoClip(make_frame_mpl, duration=dur)
    # animation.write_gif('_'.join((model_type, dataset_type, 'start_lr', str(start_lr))) + '.gif', fps=20)
    # plot fig
    make_frame_mpl(dur)
    ax.view_init(elev=90., azim=0.)
    # ax.view_init(elev=20., azim=45.)
    plt.savefig('_'.join((model_type, dataset_type, 'start_lr', str(start_lr)))+'.png')

if __name__ == '__main__':
    # utils.rm('../output  ')
    tic = time.time()
    config_dict = {'model_type': ['vgg6', 'vgg10', 'resnet6', 'resnet10'],
                   'lr': np.concatenate((np.logspace(0, -5, 6), np.logspace(-1.5, -2.5, 0))),
                   'dataset_type': ['cifar10', 'cifar100']
                   }
    # visualizer = Visualizer(config_dict, join='inner', stat_only=True, paranet_folder='stat1')
    # subplots(visualizer, path_suffix='_1')
    # subplots(visualizer, path_suffix='_1_stable_lr')

    visualizer = Visualizer(config_dict, join='inner', stat_only=True, paranet_folder='stat2')
    # subplots(visualizer, path_suffix='_2')
    # subplots(visualizer, path_suffix='_2_stable_lr')

    # visualizer = Visualizer(config_dict, join='inner', stat_only=True, paranet_folder='stat3')
    # subplots(visualizer, path_suffix='_3')
    # subplots(visualizer, path_suffix='_3_stable_lr')

    print time.time() - tic

    dataset, model_type = 'cifar100', 'resnet10'

    for lr in [0, 2, 4]:
        t_sne(visualizer, model_type, dataset, lr)
    for dataset in ['cifar10', 'cifar100']:  # , 'cifar100'
        for model_type in ['vgg6', 'resnet6', 'vgg10', 'resnet10', ]:  # 'vgg8', 'resnet8',
            t_sne(visualizer, model_type, dataset, 2)

