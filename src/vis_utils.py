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

try:
  from moviepy.video.io.bindings import mplfig_to_npimage
  import moviepy.editor as mpy
except:
  pass

from matplotlib.ticker import FormatStrFormatter

matplotlib.style.use('ggplot')

Axes3D


def drop_level(perf_df, other_name=None):
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
      if (levels_len != np.ones_like(levels_len)).all(): break
      
      for ind, level in enumerate(perf_df.columns.levels):
        if len(level) == 1:
          res_str += str(level[0]) + '_'
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


def split_level(columns, mid=2):
  df_tuples = [c.split('/') for c in columns]
  df_tuples = [['/'.join(c[:mid]), '/'.join(c[mid:])] for c in df_tuples]
  fcolumns = pd.MultiIndex.from_tuples(df_tuples, names=['layer', 'stat'])
  return fcolumns


def append_level(columns, name, value='_'):
  df_tuples = [c + (value,) for c in columns]
  levels, names, name2level, name2ind = get_columns_alias(columns)
  fname = copy.deepcopy(names)
  fname.append(name)
  
  fcolumns = pd.MultiIndex.from_tuples(df_tuples, names=fname)
  
  return fcolumns


def split_layer_stat(df):
  df, hyper_str = drop_level(df)
  
  df_name = df
  indexf = MultiIndexFacilitate(df_name.unstack().index)
  level_name = 'name'
  df_name = df_name.unstack().reset_index().pivot_table(values=0, index=level_name,
                                                        columns=set(indexf.names) - {level_name})
  
  df_name.index = split_level(df_name.index)
  level_name = 'iter'
  indexf = MultiIndexFacilitate(df_name.unstack([-2, -1]).index)
  df_name = df_name.unstack([-2, -1]).reset_index().pivot_table(values=0, index=level_name,
                                                                columns=set(indexf.names) - {level_name})
  return df_name


def custom_sort(columns, how='layer'):
  new_index0 = ['act', 'kernel', 'bias']
  new_index1 = ['diff', 'stdtime', 'iqr', 'std', 'mean', 'median', 'magmean', 'posmean', 'negmean', 'posproportion',
                'max', 'min', 'orthogonality', 'sparsity', 'ptrate-thresh-0.2', 'ptrate-thresh-0.6',
                'ptrate-thresh-mean', 'totvar']
  new_index = cartesian([new_index0, new_index1])
  new_index = ['/'.join(index_) for index_ in new_index]
  
  sort_order = {index: ind for ind, index in enumerate(new_index)}
  
  new_index0 = ['layer' + str(ind) for ind in range(100)]  # todo increase it when needed
  new_index1 = ['input', 'conv', 'bn', 'fc', 'softmax']  # although it is not necessaryli right
  new_index = cartesian([new_index0, new_index1]).tolist()
  new_index = ['/'.join(index_) for index_ in new_index]
  
  sort_order2 = {index: ind for ind, index in enumerate(new_index)}
  
  # print sort_order2
  columns = list(columns)
  if how == 'layer':
    columns.sort(key=lambda val: sort_order2[val])
  else:
    columns.sort(key=lambda val: sort_order[val])
  
  return columns


def reindex(t):
  indexf = MultiIndexFacilitate(t.unstack().index)
  dest = indexf.names2levels['layer']
  dest = custom_sort(dest)
  t = t.transpose().reindex(dest, level='layer').transpose()
  
  dest = indexf.names2levels['stat']
  dest = custom_sort(dest, how='stat')
  t = t.transpose().reindex(dest, level='stat').transpose()
  
  return t


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
  if len(arrays) == 0:
    return []
  arrays = [np.asarray(x) for x in arrays]
  # dtype = arrays[0].dtype
  dtype = object
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
  def __init__(self, paranet_folder, join='outer', stat_only=True):
    self.aggregate(join, stat_only=stat_only, parant_folder=paranet_folder)
    self.split()
    levels, names, name2level, name2ind = get_columns_alias(self.df.columns)
    self.name2level = name2level
  
  def split(self):
    self.perf_df = select(self.df, {'name': "(?:val_loss|loss|val_acc|acc)"})
    self.stat_df = select(self.df, {'name': "(?:^layer.*|^layer.*)"})
  
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
    self.columns = index
    df.index.name = 'epoch' if not stat_only else 'iter'
    self.df = df


class MultiIndexFacilitate(object):
  def __init__(self, columns):
    levels = list(columns.levels)
    names = list(columns.names)
    
    self.levels = levels
    self.names = names
    self.labels = columns.labels
    self.names2levels = {name: level for name, level in zip(names, levels)}
    self.names2ind = {name: ind for ind, name in enumerate(names)}
    self.names2len = {name: len(level) for name, level in zip(names, levels)}
    self.index = columns
  
  def update(self):
    self.index = pd.MultiIndex.from_product(self.levels, names=self.names)


def plot(perf_df, axes_names, legend=True, sup_title=''):
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
  if not utils.get_config('dbg'):
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


def select(df, level2pattern, regexp=True):
  df_name = df
  for level_name, pattern_name in level2pattern.iteritems():
    indexf = MultiIndexFacilitate(df.unstack().index)
    
    df_name = df_name.unstack().reset_index().pivot_table(values=0, index=level_name,
                                                          columns=set(indexf.names) - {level_name})
    
    if regexp:
      df_name = df_name.filter(regex=pattern_name, axis=0)
    else:
      df_name = df_name.loc[(pattern_name,), :]
    level_name = 'iter'
    df_name = df_name.unstack().reset_index().pivot_table(values=0, index=level_name,
                                                          columns=set(indexf.names) - {level_name})
  return df_name


def exclude(df, level2pattern, regexp=True):
  df_name = df
  
  for level_name, pattern_name in level2pattern.iteritems():
    indexf = MultiIndexFacilitate(df.unstack().index)
    df_name = df_name.unstack().reset_index().pivot_table(values=0, index=level_name,
                                                          columns=set(indexf.names) - {level_name})
    
    names = df_name.index
    
    all_ind = np.arange(len(names))
    if regexp:
      match_ind = [ind for ind, name in enumerate(names) if re.match(pattern_name, name)]
    else:
      match_ind = [ind for ind, name in enumerate(names) if pattern_name == name]
    
    left_ind = np.setdiff1d(all_ind, match_ind)
    df_name = df_name.iloc[left_ind, :]
    level_name = 'iter'
    df_name = df_name.unstack().reset_index().pivot_table(values=0, index=level_name,
                                                          columns=set(indexf.names) - {level_name})
  return df_name


def auto_plot(df, axes_names, path_suffix='default', ipython=True, show=False):
  df, sup_title = drop_level(df)
  if axes_names[-1] == '_':
    df.columns = append_level(df.columns,'_')
  indexf = MultiIndexFacilitate(df.columns)
  
  other_names = np.setdiff1d(indexf.names, axes_names)
  
  paths = []
  
  if len(other_names) == 0:
    _df = df.copy()
    fig, sup_title = plot(_df, axes_names, sup_title)
    utils.mkdir_p(Config.output_path + path_suffix + '/')
    sav_path = (Config.output_path
                + path_suffix + '/'
                + re.sub('/', '', sup_title)
                + '.pdf').strip('_')
    fig.savefig(sav_path,
                bbox_inches='tight')
    paths.append(sav_path)
    plt.close()
    
  for poss in cartesian([indexf.names2levels[name] for name in other_names]):
    _df = df.copy()
    for _name, _poss in zip(other_names, poss):
      _df = select(_df, {_name: _poss}, regexp=False)
    # if _df is None: break
    # if _df is None: continue
    
    if 'layer' in _df.columns.names:
      _df = reindex(_df)
    
    fig, sup_title = plot(_df, axes_names, sup_title)
    utils.mkdir_p(Config.output_path + path_suffix + '/')
    sav_path = (Config.output_path
                + path_suffix + '/'
                + re.sub('/', '', sup_title)
                + '.pdf').strip('_')
    fig.savefig(sav_path,
                bbox_inches='tight')
    paths.append(sav_path)
    if show: plt.show()
    plt.close()
    if osp.exists('dbg'):
      logger.info('dbg mode break')
      break
  
  return paths


def heatmap(paranet_folder):  # 'stat301'
  visualizer = Visualizer(join='outer', stat_only=True, paranet_folder=paranet_folder)
  df = visualizer.stat_df.copy()
  
  df.columns = split_level(df.columns)
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
    df = select(df, {'stat'   : 'act/ptrate-thresh-mean',
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


def map_name(names):
  if isinstance(names, basestring):
    names = [names]
  name_dict = {
    'conv2d'                     : 'conv',
    'dense'                      : 'fc',
    '_win_size_(.*)$'            : '/winsize-\g<1>',
    '_win_size_(\d+)_thresh_(.*)': '-thresh-\g<2>/winsize-\g<1>',
    'batchnormalization'         : 'bn'
  }
  for ind, name in enumerate(names):
    new_name = name
    for pattern, replace in name_dict.iteritems():
      new_name = re.sub(pattern, replace, new_name)
    
    names[ind] = new_name
  
  return names


if __name__ == '__main__':
  visualizer = Visualizer(paranet_folder='all')
  df = visualizer.stat_df.copy()
  
  df = select(df, {'model_type': 'vgg16'})
  df = exclude(df, {'name': '.*example.*'})
  df = split_layer_stat(df)
  
  t = select(df, {'dataset_type': 'cifar10', 'lr': '1.00e-03'}, regexp=False)
  t = exclude(t, {'layer': '.*bn.*'})
  t = exclude(t, {'layer': '.*input.*'})
  
  auto_plot(t, axes_names=('layer', 'stat', 'with_bn'))
  
  t = select(df, {'dataset_type': 'cifar10', 'lr': '1.00e-02'}, regexp=False)
  t = exclude(t, {'layer': '.*bn.*'})
  t = exclude(t, {'layer': '.*input.*'})
  t.head()
  
  auto_plot(t, axes_names=('layer', 'stat', 'with_bn'), path_suffix='default/lr1e-2')
  
  t = select(df, {'dataset_type': 'cifar10', 'lr': '1.00e-03'}, regexp=False)
  # t=exclude(t,{'layer':'.*bn.*'})
  t = exclude(t, {'layer': '.*input.*'})
  t.head()
  
  auto_plot(t, axes_names=('layer', 'stat', 'with_bn'), path_suffix='default/seebn')
  
  t = select(df, {'dataset_type': 'cifar10', 'lr': '1.00e-03', 'with_bn': 'False'}, regexp=False)
  t = exclude(t, {'layer': '.*input.*'})
  auto_plot(t, axes_names=('layer', 'stat', '_'), path_suffix='default/nobn')
