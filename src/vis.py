import matplotlib

matplotlib.use('Agg')

from loader import Loader
from logs import logger
from configs import Config
from datasets import Dataset
from models import VGG

from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from matplotlib.ticker import FormatStrFormatter

matplotlib.style.use('ggplot')

Axes3D


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


def merge_level(columns, start, stop):
    columns_tuples = [c[:start] + (('/'.join(c[start:stop])).strip('/'),) + c[stop:] for c in columns]
    return pd.MultiIndex.from_tuples(columns_tuples)


def expand_level(columns):
    df_tuples = [('/'.join(c).split('/')) for c in columns]
    df_tuples_len = np.array([len(_tmp) for _tmp in df_tuples]).max()
    levels, names, name2level, name2ind = get_columns_alias(columns)
    fname = copy.deepcopy(names[:-1])

    for _ind in range(df_tuples_len - len(fname)):
        fname.append('name' + str(_ind))
    fcolumns = pd.MultiIndex.from_tuples([('/'.join(c).split('/')) for c in columns], names=fname)

    tmp = np.array([list(_tmp) for _tmp in fcolumns])
    tmp[tmp == float('nan') or tmp == 'nan'] = ''
    tmp = tmp.astype(basestring)
    fcolumns = pd.MultiIndex.from_arrays(tmp.transpose(), names=fname)
    return fcolumns


@utils.static_vars(ind=0, label2color={})
def get_colors(label):
    colors = [u'#E24A33', u'#348ABD', u'#988ED5', u'#777777', u'#FBC15E', u'#8EBA42', u'#FFFF00', u'#FF8C00',
              u'#FFEFD5', u'#FFA500', u'#6B8E23', u'#87CEEB', u'#006400', u'#008080',
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


def get_columns_alias(columns):
    levels = columns.levels
    names = columns.names
    name2level = {name: level for name, level in zip(names, levels)}
    name2ind = {name: ind for ind, name in enumerate(names)}
    return list(levels), list(names), name2level, name2ind


class Visualizer(object):
    def __init__(self, paranet_folder, join='inner', stat_only=True, ):
        self.aggregate(join, stat_only=stat_only, parant_folder=paranet_folder)
        self.split()
        levels, names, name2level, name2ind = get_columns_alias(self.df.columns)
        self.name2level = name2level

    def split(self):
        self.perf_df = select(self.df, {'name': "(?:val_loss|loss|val_acc|acc)"})
        self.stat_df = select(self.df, {'name': "(?:^obs.*|^layer.*)"})

    def aggregate(self, join, parant_folder, stat_only):
        conf_name_dict = {}
        loaders = {}
        parant_path = Config.root_path + '/' + parant_folder + '/'
        for path in glob.glob(parant_path + '/*'):
            _conf = utils.unpickle(path + '/config.pkl')
            loader = Loader(path=path, stat_only=stat_only)
            # loader.start()
            loader.load(stat_only=stat_only)
            loaders[_conf.name] = loader
            conf_name_dict[_conf.name] = _conf.to_dict()

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
                # names = conf_dict.values() + [name]
                # names = map_name(names)
                name = map_name(name)[0]
                index_l.append(conf_dict.values() + [name])

            df_l.append(act)
            for name in act.columns:
                name = map_name(name)[0]
                index_l.append(conf_dict.values() + [name])

            df_l.append(param)
            for name in param.columns:
                name = map_name(name)[0]
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


def plot(perf_df, axes_names, other_names=None, legend=True):
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
    if not osp.exists('dbg'):
        figsize = (4.2 * cols, 2.25 * rows)
    else:
        logger.info('dbg small fig mode')
        figsize = (4.2, 2.25)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # plt.tight_layout(pad=3, w_pad=.9, h_pad=1.5,rect=(.2,.2,1,1))
    fig.subplots_adjust(hspace=1., wspace=.6)
    axes = np.array(axes).reshape(rows, cols)

    # for _i in range(rows):
    #     axes[_i][0].set_ylabel(re.sub('/','\n',row_level[_i]),rotation=0, fontsize=15,labelpad=35)
    #
    # for _j in range(cols):
    #     axes[0][_j].set_title(col_level[_j], y=1.09)

    for _i in range(rows):
        for _j in range(cols):
            axes[_i][_j].set_ylabel(re.sub('/', '\n', row_level[_i]), rotation=0, fontsize=13, labelpad=36)
            axes[_i][_j].set_title(col_level[_j], y=1.09)

    # # plot to right place
    target = []
    legends = np.empty((rows, cols)).astype(object)
    for _row in range(rows):
        for _col in range(cols):
            legends[_row, _col] = []

    for inds in perf_df.columns:
        for ind in inds:
            if ind in row_level: _row = level2row[ind]
            if ind in col_level: _col = level2col[ind]
        for ind in inds:
            if ind in inside_level:
                if legends[_row, _col] == []:
                    legends[_row, _col] = [str(ind)]
                else:
                    legends[_row, _col] += [str(ind)]
        target.append(axes[_row, _col])

    # perf_df.plot(subplots=True, legend=False, ax=target, marker=None, sharex=False)
    perf_df.interpolate().plot(subplots=True, legend=False, ax=target, marker=None, sharex=False)
    #  # change color

    for axis in axes.flatten():
        for line in axis.get_lines():
            _label = line.get_label()
            for level in inside_level:
                if level in _label: label = level
            color = get_colors(label)
            line.set_color(color)

    # # plot legend
    # if legend:
    #     axes[0, 0].legend(list(legends[0, 0]))
    # legend_set=set()
    for _row in range(legends.shape[0]):
        for _col in range(legends.shape[1]):
            axes[_row, _col].yaxis.get_major_formatter().set_powerlimits((-2, 2))
            _ylim = axes[_row, _col].get_ylim()
            if np.diff(_ylim) < 1e-7 and np.mean(_ylim) > 1e-3:
                logger.info('Attatin: float error' + str(_ylim) + str((_row, _col)))
                if _row == legends.shape[0] - 1:
                    axes[_row, _col].set_ylim([0, 1])
            # if legend:
            # try:
            if len(legends[_row, _col]) > 1 and _row == 0:
                axes[_row, _col].legend(legends[_row, _col])
                # logger.info('legend' + str(legends[_row, _col]))
            else:
                # logger.debug('shouldnot has legend')
                axes[_row, _col].legend([])

                # except:
                #     from IPython import  embed;embed()
                # else:
                #     axes[_row, _col].legend([])

    sup_title += 'legend_' + inside
    sup_title = sup_title.strip('_')
    fig.suptitle(sup_title, fontsize=50)
    # plt.show()
    return fig, sup_title


def select(df, level2pattern, sort_names=None, regexp=True):
    df = df.copy()
    for level_name, pattern_name in level2pattern.iteritems():
        sel_name = df.columns.get_level_values(level_name)
        f_sel_name = set()
        pat = re.compile(pattern_name)
        for sel_name_ in sel_name:
            judge = bool(pat.match(sel_name_)) if regexp else pattern_name == sel_name_
            if judge: f_sel_name.add(sel_name_)
        df_l = []
        for sel_name_ in f_sel_name:
            df_l.append(df.xs(sel_name_, level=level_name, axis=1, drop_level=False))
        if df_l != []:
            df = pd.concat(df_l, axis=1)
        else:
            return None
        if sort_names is None: sort_names = df.columns.names
        # df.sort_index(level=sort_names, axis=1, inplace=True)
    return df


def auto_plot(df, path_suffix, axes_names, other_names):
    columns = df.columns
    levels, names, name2level, name2ind = get_columns_alias(columns)
    show = False
    pathss = []
    for poss in cartesian([name2level[name] for name in other_names]):
        _df = df.copy()
        for _name, _poss in zip(other_names, poss):
            _df = select(_df, {_name: _poss}, regexp=False)
            if _df is None: break
        if _df is None: continue

        if _df.columns.names[2] == 'stat':
            _df = reindex(_df, level=2)

        fig, sup_title = plot(_df, axes_names, other_names)
        utils.mkdir_p(Config.output_path + path_suffix + '/')
        sav_path = (Config.output_path
                    + path_suffix + '/'
                    + re.sub('/', '', sup_title)
                    + '.pdf').strip('_')
        fig.savefig(sav_path,
                    bbox_inches='tight')
        pathss.append(sav_path)
        if show: plt.show()
        plt.close()
        if osp.exists('dbg'):
            logger.info('dbg mode break')
            break
    return pathss


def append_level(columns, name, value=''):
    df_tuples = [c + (value,) for c in columns]
    levels, names, name2level, name2ind = get_columns_alias(columns)
    fname = copy.deepcopy(names)
    fname.append(name)

    fcolumns = pd.MultiIndex.from_tuples(df_tuples, names=fname)

    return fcolumns


def reindex(df, level):
    new_index0 = ['act/', 'kernel/', 'bias/']
    new_index1 = ['diff', 'stdtime', 'iqr', 'std', 'mean', 'median', 'magmean', 'posmean', 'negmean', 'posproportion',
                  'max', 'min', 'orthogonality', 'sparsity', 'ptrate-thresh-0.2', 'ptrate-thresh-0.6',
                  'ptrate-thresh-mean', 'totvar']
    new_index = [ind0 + ind1 for ind0 in new_index0 for ind1 in new_index1]
    diff1 = ['kernel/', 'bias/']
    diff2 = ['ptrate-thresh-0.2', 'ptrate-thresh-0.6', 'ptrate-thresh-mean']
    diff = [_diff1 + _diff2 for _diff1 in diff1 for _diff2 in diff2] + ['bias/orthogonality']
    new_index = [_new_index for _new_index in new_index if _new_index not in diff ]

    df = df.transpose().reindex(new_index, level=level).transpose().copy()
    return df


def subplots(visualizer, path_suffix):
    perf_df = visualizer.perf_df.copy()
    # plot only val acc
    # perf_df = select(perf_df, {'name': 'val_acc$'})
    perf_df.columns = merge_level(perf_df.columns, start=0, stop=3)
    perf_df.columns = append_level(perf_df.columns, '')
    perf_df.columns = append_level(perf_df.columns, 't2')
    perf_df.columns.set_names(['hyper', 'metric', '', 't2'], inplace=True)
    auto_plot(perf_df, path_suffix + '_perf',
              axes_names=['hyper', 'metric', ''],
              other_names=['t2'])

    # plot statistics
    df = visualizer.stat_df.copy()

    df.columns = expand_level(df.columns)
    # # name0 1 2 3 -- obs0 conv2d act iqr
    # df = select(df, {'dataset_type': 'cifar10'})
    df.columns = merge_level(df.columns, start=5, stop=7)
    df = reindex(df, 5)
    df.columns = merge_level(df.columns, start=3, stop=5)

    df.columns = merge_level(df.columns, start=0, stop=3)
    df.columns.set_names(['hyper', 'layer', 'stat', 'winsize'], inplace=True)
    df = select(df,
                {'hyper': '(?:resnet10/cifar10/1.*3|vgg10/cifar10/1.*2)'})  # (?:resnet10/cifar10/.*3|vgg10/cifar10/.*2)

    paths = auto_plot(df, path_suffix + '_stat',
                      axes_names=('layer', 'stat', 'winsize'),
                      other_names=['hyper'])
    utils.merge_pdf(paths)


def heatmap(paranet_folder):  # 'stat301'
    visualizer = Visualizer(join='outer', stat_only=True, paranet_folder=paranet_folder)
    df = visualizer.stat_df.copy()

    df.columns = expand_level(df.columns)
    df.columns = merge_level(df.columns, start=7, stop=10)
    df.columns = merge_level(df.columns, start=5, stop=7)
    df.columns = merge_level(df.columns, start=3, stop=5)
    df.columns = merge_level(df.columns, start=0, stop=3)
    df.columns.set_names(['hyper', 'layer', 'stat', 'winsize'], inplace=True)
    levels, names, name2level, name2ind = get_columns_alias(df.columns)

    df_ori = df.copy()
    limits = None
    pathss = []
    for ind, name in enumerate(list(name2level['hyper'])):
        fig, ax = plt.subplots(1, figsize=(5, 40))
        df = select(df_ori, {'hyper': name}, regexp=False)
        df = select(df, {'stat': 'act/ptrate-thresh-mean',
                         'winsize': 'winsize-31'})
        df2, suptitle = drop_level(df, ['hyper', 'winsize', 'stat'])
        suptitle = re.sub('/', '_', suptitle).strip('_')

        if limits == None:
            limits = df2.values.transpose().shape[1]
        df2.sort_index(axis=1, inplace=True)
        mat = df2.values.transpose()[:10, :limits]
        if np.isnan(mat).any():
            logger.warning('mat conations nan' + suptitle)
        ax.set_title(suptitle, fontsize=5)
        im = ax.imshow(mat, interpolation='none')
        # ax.set_aspect(1.5)
        #
        # ax.tick_params(axis=u'both', which=u'both', length=0)
        # ax.set_xlim(-0.5, limits-0.5)
        # ax.set_ylim(10.5, -0.5)
        ax.grid('off')
        plt.xticks(rotation=20)
        xticks = np.full((limits,), '').astype(basestring)
        xticks[::5] = np.array(df2.index[:limits][::5]).astype(basestring)
        ax.set_xticklabels(xticks)
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1%", pad=0.1)

        cbar = plt.colorbar(im, cax=cax)
        print mat[~np.isnan(mat)].min(), mat[~np.isnan(mat)].max()
        # cbar.ax.get_yaxis().set_ticks([0,1])
        cbar.ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.2f'))
        paths = Config.root_path + '/' + suptitle + '.pdf'
        fig.savefig(paths, bbox_inches='tight')  # bbox_inches='tight'
        pathss.append(paths)
    utils.merge_pdf(pathss)
    return pathss


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
        logger.debug('{}'.format(_perf_df.columns))
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
    plt.savefig(parant_folder + '_'.join((model_type, dataset_type, 'start_lr', str(start_lr))) + '.pdf')


def map_name(names):
    if isinstance(names, basestring):
        names = [names]
    name_dict = {'obs': 'layer',
                 'conv2d': 'conv',
                 'dense': 'fc',
                 '_win_size_(.*)$': '/winsize-\g<1>',
                 '_win_size_(\d+)_thresh_(.*)': '-thresh-\g<2>/winsize-\g<1>',
                 }
    for ind, name in enumerate(names):
        new_name = name
        for pattern, replace in name_dict.iteritems():
            new_name = re.sub(pattern, replace, new_name)

        names[ind] = new_name

    return names


if __name__ == '__main__':
    tic = time.time()
    for parant_folder in ['stat301_10']:
        visualizer = Visualizer(join='outer', stat_only=True, paranet_folder=parant_folder)
        subplots(visualizer, path_suffix=parant_folder.strip('stat'))
        heatmap(parant_folder)
    print time.time() - tic

    # dataset, model_type = 'cifar100', 'resnet10'
    #
    # for lr in [0, 2, 4]:
    #     t_sne(visualizer, model_type, dataset, lr)
    # for dataset in ['cifar10', 'cifar100']:  # , 'cifar100'
    #     for model_type in ['vgg6', 'resnet6', 'vgg10', 'resnet10', ]:  # 'vgg8', 'resnet8',
    #         t_sne(visualizer, model_type, dataset, 2)
