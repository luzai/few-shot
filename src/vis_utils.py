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
from utils import cartesian, dict2df
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


def drop_level(perf_df, allow_downcast=False):
  indexf = MultiIndexFacilitate(perf_df.columns)
  
  res_str = ''
  other_name = np.array(indexf.names)[np.array(indexf.names2len.values()) == 1].tolist()
  perf_df.columns = perf_df.columns.droplevel(other_name)
  
  for name in other_name:
    level = indexf.names2levels[name]
    # print name, level
    res_str += level[0] + '_'
  if not isinstance(perf_df.columns, pd.MultiIndex) and not allow_downcast:
    perf_df.columns = append_level(perf_df.columns, '_')
  return perf_df, res_str


def merge_level(columns, start, stop):
  columns_tuples = [c[:start] + (('/'.join(c[start:stop])).strip('/'),) + c[stop:] for c in columns]
  return pd.MultiIndex.from_tuples(columns_tuples)


def split_level(columns, mid=2):
  df_tuples = [c.split('/') for c in columns]
  df_tuples = [['/'.join(c[:mid]), '/'.join(c[mid:])] for c in df_tuples]
  fcolumns = pd.MultiIndex.from_tuples(df_tuples, names=['layer', 'stat'])
  return fcolumns


def append_level(columns, name='_', value='__'):
  df_tuples = []
  for c in columns:
    if isinstance(c, tuple):
      df_tuples += [c + (value,)]
    else:
      df_tuples += [(c,) + (value,)]
  fname = copy.deepcopy(list(columns.names))
  fname.append(name)
  try:
    fcolumns = pd.MultiIndex.from_tuples(df_tuples, names=fname)
  except:
    print fname
    print df_tuples
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
  # todo please do not add thighs that do not exsits
  new_index0 = ['act', 'kernel', 'bias', 'beta', 'gamma', 'moving-mean', 'moving-var']
  new_index1 = ['diff', 'stdtime', 'iqr', 'std', 'mean', 'median',
                'norm', 'magmean', 'magnitude', 'posmean', 'negmean', 'posproportion',
                'max', 'min', 'ortho', 'sparsity', 'ptrate-thresh-0.2', 'ptrate-thresh-0.6',
                'ptrate-thresh-mean', 'ptrate', 'totvar', 'updateratio',
                'ortho', 'orthoabs',
                'orthochnl', 'orthochnlabs', 'orthosmpl', 'orthosmplabs', 'orthochannel', 'orthosample'
    , 'orthogonalitychannel', 'orthogonalitysample', 'orthogonality'
                ]
  new_index = cartesian([new_index0, new_index1])
  new_index = ['/'.join(index_) for index_ in new_index]
  new_index.insert(0, 'dummy')
  sort_order = {index: ind for ind, index in enumerate(new_index)}
  
  new_index0 = ['layer' + str(ind) for ind in range(20)] + ['Layer' + str(ind) for ind in range(20)]
  new_index1 = ['input', 'conv', 'bn', 'conv-s', 'add', 'fc', 'softmax']  # although it is not necessaryli right
  new_index = cartesian([new_index0, new_index1]).tolist()
  new_index = ['/'.join(index_) for index_ in new_index]
  new_index.insert(0, 'dummy')
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
  colors = [u'#348ABD', u'#988ED5', u'#8EBA42', u'#E24A33', u'#FFA500', u'#FFFF00', u'#FBC15E', u'#777777', u'#FF8C00',
            u'#FFEFD5', u'#6B8E23', u'#87CEEB', u'#006400', u'#008080',
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


# class ColorManager

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
    # if axes_names[-1] != 'lr':
    #   continue
    # else:
    yield axes_names


def get_columns_alias(columns):
  levels = columns.levels
  names = columns.names
  name2level = {name: level for name, level in zip(names, levels)}
  name2ind = {name: ind for ind, name in enumerate(names)}
  return list(levels), list(names), name2level, name2ind


from utils import Timer

timer = Timer()


class Visualizer(object):
  def __init__(self, paranet_folder=None, join='outer'):
    if paranet_folder is not None:
      timer.tic()
      self.df = self.aggregate(join, stat_only=True, parant_folder=paranet_folder)
      self.split()
      self.stat_df = split_layer_stat(self.stat_df)
      
      self.tensor = self.aggregate(join, stat_only=False, parant_folder=paranet_folder, parallel=True)
      # self.tensor = split_layer_stat(self.tensor)
      # levels, names, name2level, name2ind = get_columns_alias(self.df.columns)
      # self.name2level = name2level
      print timer.toc()
    else:
      self.lr, self.val_acc, self.perf_df, self.stat_df = None, None, None, None,
  
  def split(self):
    self.perf_df = select(self.df, {'name': "(?:val_loss|loss|val_acc|acc)"}).dropna(how='all')
    self.stat_df = select(self.df, {'name': "(?:^Layer.*|^layer.*)"}).dropna(how='all')
    self.lr = select(self.df, {'name': 'lr'}).dropna(how='all')
    self.val_acc = select(self.df, {'name': 'val_acc'}).dropna(how='all')
  
  def aggregate(self, join, parant_folder, stat_only, parallel=True):
    conf_name_dict = {}
    loaders = {}
    parant_path = Config.root_path + '/' + parant_folder + '/'
    for path in glob.glob(parant_path + '/*'):
      _conf = utils.unpickle(path + '/config.pkl')
      if stat_only:
        if not osp.exists(path + '/miscellany'): return
      if not stat_only:
        if not osp.exists(path + '/act') or not osp.exists(path + '/param'): return
      loader = Loader(path=path, stat_only=stat_only, parallel=parallel)
      # loader.start()
      loader.load(stat_only=stat_only, parallel=parallel)
      loaders[_conf.name] = loader
      conf_name_dict[_conf.name] = _conf.to_dict()
    
    df_l = []
    index_l = []
    assert len(conf_name_dict) != 0, 'should not be empty'
    for ind in range(len(conf_name_dict)):
      conf_name = conf_name_dict.keys()[ind]
      conf_dict = conf_name_dict[conf_name]
      loader = loaders[conf_name]
      if stat_only:
        scalar = loader.scalars
        df_l.append(scalar)
      else:
        scalar = pd.concat([loader.act, loader.params], axis=1, join=join)
        df_l.append(scalar)
      for name in scalar.columns:
        name = map_name(name)[0]
        index_l.append(conf_dict.values() + [name])
        
        # act = loader.act
        # df_l.append(act)
        # for name in act.columns:
        #   name = map_name(name)[0]
        #   index_l.append(conf_dict.values() + [name])
        
        # param = loader.params
        # df_l.append(param)
        # for name in param.columns:
        #   name = map_name(name)[0]
        #   index_l.append(conf_dict.values() + [name])
    
    index_l = np.array(index_l).astype(basestring).transpose()
    index_name = conf_dict.keys() + ['name']
    index = pd.MultiIndex.from_arrays(index_l, names=index_name)
    df = pd.concat(df_l, axis=1, join=join)
    df.columns = index
    df = df.sort_index(axis=1, level=index_name)
    self.names = index_name
    self.columns = index
    df.index.name = 'iter'
    return df
  
  def copy(self):
    vis = Visualizer()
    vis.lr, vis.val_acc, vis.perf_df, vis.stat_df = self.lr.copy(), self.val_acc.copy(), self.perf_df.copy(), self.stat_df.copy()
    return vis
  
  def select(self, level2pattern, regexp=True):
    self.stat_df = select(self.stat_df, level2pattern, regexp)
    self.lr = select(self.lr, level2pattern, regexp)
    self.val_acc = select(self.val_acc, level2pattern, regexp)
  
  def exclude(self, level2pattern, regexp=True):
    self.stat_df = exclude(self.stat_df, level2pattern, regexp)
    self.lr = exclude(self.lr, level2pattern, regexp)
    self.val_acc = exclude(self.val_acc, level2pattern, regexp)
  
  def auto_plot(self, axes_names=('layer', 'stat', '_'),
                path_suffix='_default',
                ipython=True,
                show=False):
    df = self.stat_df
    lr = self.lr
    val_acc = self.val_acc
    df, sup_title = drop_level(df)
    if len(df.columns.names) < 3 or axes_names[-1] == '_':
      df.columns = append_level(df.columns, '_')
    indexf = MultiIndexFacilitate(df.columns)
    other_names = np.setdiff1d(indexf.names, axes_names)
    if axes_names[-1] != '_':
      df = append_dummy(df)
    
    paths = []
    if len(other_names) == 0:
      _df = df.copy()
      if 'stat' in _df.columns.names:
        _df = reindex(_df)
      _df = _df.dropna(how='all').dropna(how='all', axis=1)
      fig, sup_title = plot(_df, axes_names, sup_title, lr=lr, val_acc=val_acc)
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
      _lr = lr.copy()
      _val_acc = val_acc.copy()
      for _name, _poss in zip(other_names, poss):
        _df = select(_df, {_name: _poss}, regexp=False)
        _lr = select(_lr, {_name: _poss}, regexp=False)
        _val_acc = select(_val_acc, {_name: _poss}, regexp=False)
      
      sup_title_ = '_' + utils.list2str(poss) + sup_title
      
      # if _df is None: break
      # if _df is None: continue
      
      if 'layer' in _df.columns.names:
        _df = reindex(_df)
      
      fig, sup_title_ = plot(_df, axes_names, sup_title_, lr=_lr, val_acc=_val_acc)
      utils.mkdir_p(Config.output_path + path_suffix + '/')
      sav_path = (Config.output_path
                  + path_suffix + '/'
                  + re.sub('/', '', sup_title_)
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


def auto_plot(df, axes_names=('layer', 'stat', '_'),
              path_suffix='_default',
              ipython=True,
              show=False):
  df, sup_title = drop_level(df)
  if len(df.columns.names) < 3 or axes_names[-1] == '_':
    df.columns = append_level(df.columns, '_')
  indexf = MultiIndexFacilitate(df.columns)
  other_names = np.setdiff1d(indexf.names, axes_names)
  if axes_names[-1] != '_' and axes_names[0] == 'layer':
    df = append_dummy(df)
  
  paths = []
  if len(other_names) == 0:
    _df = df.copy()
    if 'stat' in _df.columns.names:
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
    plt.close()
  
  for poss in cartesian([indexf.names2levels[name] for name in other_names]):
    _df = df.copy()
    for _name, _poss in zip(other_names, poss):
      _df = select(_df, {_name: _poss}, regexp=False)
    _df = _df.dropna(how='all').dropna(how='all', axis=1)
    sup_title_ = '_' + utils.list2str(poss) + sup_title
    
    if 'layer' in _df.columns.names:
      _df = reindex(_df)
    
    fig, sup_title_ = plot(_df, axes_names, sup_title_)
    utils.mkdir_p(Config.output_path + path_suffix + '/')
    sav_path = (Config.output_path
                + path_suffix + '/'
                + re.sub('/', '', sup_title_)
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


def plot(perf_df, axes_names, sup_title, lr=None, val_acc=None, legend=True, ):
  # print perf_df.head()
  # print perf_df.columns.levels
  perf_df = perf_df.dropna(how='all').dropna(how='all', axis=1)
  # print perf_df.columns.levels
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
  # print insides
  from cycler import cycler
  
  # # arrange xlabel ylabel
  if not utils.get_config('dbg'):
    figsize = (4.2 * cols, 2.25 * rows)
  else:
    logger.info('dbg small fig mode')
    figsize = (4.2, 2.25)
  if lr is not None:
    fig, axes = plt.subplots(rows + 2, cols, figsize=figsize)
    axes = np.array(axes).reshape(rows + 2, cols)
    if insides >= 2:
      refaxis = 1
    else:
      refaxis = 0
    for _j in range(cols):
      ax = axes[refaxis, _j]
      ax.set_title('lr')
      ((lr.interpolate(limit_direction='backward') + lr.interpolate()) / 2.).plot(ax=ax, legend=False)
      ax = axes[refaxis + 1, _j]
      ax.set_title('val_acc')
      ((val_acc.interpolate(limit_direction='backward') + val_acc.interpolate()) / 2.).plot(ax=ax, legend=False)
    
    for axis in axes[(refaxis, refaxis + 1), :].flatten():
      for line in axis.get_lines():
        _label = line.get_label()
        _label = [__label.strip('(').strip(')') for __label in _label.split(', ')]
        label = None
        for __label in _label:
          if __label in inside_level:
            label = __label
        # print label
        color = get_colors(label)
        line.set_color(color)
    if insides >= 2:
      axes = np.concatenate([axes[(refaxis - 1,), :], axes[refaxis + 2:, :]])
    else:
      axes = axes[2:, :]
  else:
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(rows, cols)
  
  # plt.tight_layout(pad=3, w_pad=.9, h_pad=1.5,rect=(.2,.2,1,1))
  fig.subplots_adjust(hspace=1., wspace=.6)
  
  for _i in range(rows):
    for _j in range(cols):
      row_name = re.sub('/', '\n', row_level[_i])
      row_name = row_name.strip('_')
      row_name = re.sub('_', '\n', row_name)
      axes[_i][_j].set_ylabel(row_name, rotation=0, fontsize=13, labelpad=36)
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
  
  perf_df_inter = (perf_df.interpolate(limit_direction='backward') + perf_df.interpolate()) / 2.
  
  perf_df_inter.plot(subplots=True, legend=False,
                     ax=target,
                     sharex=False,
                     # style='.'
                     )
  #  # change color
  
  for axis in axes.flatten():
    for line in axis.get_lines():
      _label = line.get_label()
      _label = [__label.strip('(').strip(')') for __label in _label.split(', ')]
      label = None
      for __label in _label:
        if __label in inside_level:
          label = __label
      # print label
      color = get_colors(label)
      line.set_color(color)
  
  for _row in range(legends.shape[0]):
    for _col in range(legends.shape[1]):
      axes[_row, _col].yaxis.get_major_formatter().set_powerlimits((-2, 2))
      _ylim = axes[_row, _col].get_ylim()
      if np.diff(_ylim) < 1e-7 and np.mean(_ylim) > 1e-3:
        logger.info('Attatin: float error' + str(_ylim) + str((_row, _col)))
        if _row == legends.shape[0] - 1:
          axes[_row, _col].set_ylim([0, 1])
      
      if len(legends[_row, _col]) > 1 and _row == 0:
        axes[_row, _col].legend(legends[_row, _col])
        # logger.info('legend' + str(legends[_row, _col]))
      else:
        # logger.debug('shouldnot has legend')
        axes[_row, _col].legend([])
  
  # from IPython import embed;embed()
  sup_title += '_' + inside
  sup_title = sup_title.strip('_')
  fig.suptitle(sup_title, fontsize=25)
  # plt.show()
  return fig, sup_title


def interpolate(perf_df):
  return (perf_df.interpolate(limit_direction='backward') + perf_df.interpolate()) / 2.


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


def append_dummy(df):
  indexf = MultiIndexFacilitate(df.unstack().index)
  level_name = 'layer'
  t = df.unstack().reset_index().pivot_table(values=0, index=level_name,
                                             columns=set(indexf.names) - {level_name})
  t.head()
  t.loc['dummy', :] = 1
  
  level_name = 'iter'
  df_name = t.unstack().reset_index().pivot_table(values=0, index=level_name,
                                                  columns=set(indexf.names) - {level_name})
  
  return df_name


def resample(df):
  interval = np.diff(df.index).max()
  new_index = np.concatenate([df.index, range(min(df.index), max(df.index), interval)])
  new_index.sort()
  new_index = np.unique(new_index)
  df = df.reindex(new_index).interpolate()
  df = df.loc[range(min(df.index), max(df.index) + 1, interval), :]
  return df


def heatmap(df):
  df, super_title = drop_level(df)
  
  indexf = MultiIndexFacilitate(df.unstack().index)
  dest = indexf.names2levels['layer']
  dest = custom_sort(dest)
  df = df.transpose().reindex(dest, level='layer').transpose()
  
  df = resample(df)
  
  mat = df.values.transpose()
  
  fig, ax = plt.subplots(1, figsize=(50, 50))
  im = ax.imshow(mat, interpolation='none')
  ax.grid('off')
  plt.xticks(rotation=20)
  ax.set_title(super_title)
  xticks = np.full((mat.shape[1],), '').astype(basestring)
  xticks[::10] = np.array(df.index[::10]).astype(basestring)
  ax.set_xticklabels(xticks)
  ax.set_xlim(-1.5, len(xticks) + 0.5)
  
  yticks = np.array([' ', ] + list(df.columns.levels[0]))
  ax.set_yticklabels(yticks)
  ax.set_ylim(len(yticks) - 1.5, -.5)
  
  import matplotlib.ticker as ticker
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="1%", pad=0.1)
  
  cbar = plt.colorbar(im, cax=cax)
  # print mat[~np.isnan(mat)].min(), mat[~np.isnan(mat)].max()
  # cbar.ax.get_yaxis().set_ticks([0,1])
  cbar.ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.2f'))
  paths = Config.root_path + '/' + re.sub('/', '_', super_title) + '.pdf'
  fig.savefig(paths, bbox_inches='tight')  # bbox_inches='tight'
  return paths


def auto_heatmap(df):
  # df = split_layer_stat(df)
  indexf = MultiIndexFacilitate(df.columns)
  other_names = np.setdiff1d(indexf.names, ['layer', 'stat'])
  paths = []
  for poss in cartesian([indexf.names2levels[name] for name in other_names]):
    _df = df.copy()
    for _name, _poss in zip(other_names, poss):
      _df = select(_df, {_name: _poss}, regexp=False)
    _df = select(_df, {'stat': 'act/ptrate'})
    _df = _df.dropna(how='all').dropna(how='all', axis=1)
    _df = _df.iloc[21:, :]
    print _df.shape
    paths.append(heatmap(_df))
  
  return paths


def map_name(names):
  if isinstance(names, basestring):
    names = [names]
  name_dict = {
    'conv2d'                        : 'conv',
    'dense'                         : 'fc',
    # '_win_size_(.*)$'            : '/winsize-\g<1>',
    # '_win_size_(\d+)_thresh_(.*)': '-thresh-\g<2>/winsize-\g<1>',
    'ptrate_win_size_11_thresh_mean': 'ptrate',
    'totvar_win_size_11'            : 'totvar',
    'batchnormalization'            : 'bn',
    # 'ortho'                         : 'ortho-',
    # 'abs'                           : '-abs'
  }
  for ind, name in enumerate(names):
    new_name = name
    for pattern, replace in name_dict.iteritems():
      new_name = re.sub(pattern, replace, new_name)
    names[ind] = new_name
  
  return names


import logs

if __name__ == '__main__':
  logger.setLevel(logs.WARN)
  visualizer = Visualizer(paranet_folder='all')
  df = visualizer.stat_df.copy()
  
  vis = visualizer.copy()
  # vis.stat_df = split_layer_stat(vis.stat_df)
  vis.select({'model_type': 'vgg10', 'optimizer': '_lr_0.001_name_sgd'}, regexp=False)
  vis.stat_df = select(vis.stat_df, {'stat': '.*orth.*'})
  vis.stat_df = exclude(vis.stat_df, {'layer': '.*bn.*'})
  vis.auto_plot(('layer', 'stat', '_'))
  
  df = visualizer.stat_df.copy()
  # df = df.iloc[:, :200]
  df = split_layer_stat(df)
  df = select(df, {'model_type': 'vgg10'}, regexp=False)
  df = exclude(df, {'optimizer': '.*001.*'})
  
  lr = visualizer.lr.copy()
  val_acc = visualizer.val_acc.copy()
  lr = select(lr, {'model_type': 'vgg10'}, regexp=False)
  lr = exclude(lr, {'optimizer': '.*001.*'})
  val_acc = select(val_acc, {'model_type': 'vgg10'}, regexp=False)
  val_acc = exclude(val_acc, {'optimizer': '.*001.*'})
  df.head()
  
  # auto_plot(df,lr,val_acc,('layer','stat','_'))
  auto_plot(df, lr, val_acc, ('layer', 'stat', 'optimizer'))
  
  df = visualizer.stat_df.copy()
  # df = df.iloc[:, :200]
  df = split_layer_stat(df)
  df = select(df, {'model_type': 'resnet10'}, regexp=False)
  df = exclude(df, {'optimizer': '.*001.*'})
  
  lr = visualizer.lr.copy()
  val_acc = visualizer.val_acc.copy()
  lr = select(lr, {'model_type': 'resnet10'}, regexp=False)
  lr = exclude(lr, {'optimizer': '.*001.*'})
  
  val_acc = select(val_acc, {'model_type': 'resnet10'}, regexp=False)
  val_acc = exclude(val_acc, {'optimizer': '.*001.*'})
  df.head()
  
  auto_plot(df, lr, val_acc, ('layer', 'stat', 'optimizer'))
